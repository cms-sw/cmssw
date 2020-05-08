#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerPerSL.h"
#include <cmath>

using namespace edm;
using namespace std;
using namespace cmsdt;
// ============================================================================
// Constructors and destructor
// ============================================================================
MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : MuonPathAnalyzer(pset, iC),
      bxTolerance_(30),
      minQuality_(LOWQGHOST),
      chiSquareThreshold_(50),
      debug_(pset.getUntrackedParameter<bool>("debug")),
      chi2Th_(pset.getUntrackedParameter<double>("chi2Th")),
      tanPhiTh_(pset.getUntrackedParameter<double>("tanPhiTh")),
      use_LSB_(pset.getUntrackedParameter<bool>("use_LSB")),
      tanPsi_precision_(pset.getUntrackedParameter<double>("tanPsi_precision")),
      x_precision_(pset.getUntrackedParameter<double>("x_precision")) {
  if (debug_)
    cout << "MuonPathAnalyzer: constructor" << endl;

  setChiSquareThreshold(chi2Th_ * 100.);

  //shift
  int rawId;
  shift_filename_ = pset.getParameter<edm::FileInPath>("shift_filename");
  std::ifstream ifin3(shift_filename_.fullPath());
  double shift;
  if (ifin3.fail()) {
    throw cms::Exception("Missing Input File")
        << "MuonPathAnalyzerPerSL::MuonPathAnalyzerPerSL() -  Cannot find " << shift_filename_.fullPath();
  }
  while (ifin3.good()) {
    ifin3 >> rawId >> shift;
    shiftinfo_[rawId] = shift;
  }

  chosen_sl_ = pset.getUntrackedParameter<int>("trigger_with_sl");

  if (chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4) {
    std::cout << "chosen sl must be 1,3 or 4(both superlayers)" << std::endl;
    assert(chosen_sl_ != 1 && chosen_sl_ != 3 && chosen_sl_ != 4);  //4 means run using the two superlayers
  }

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

MuonPathAnalyzerPerSL::~MuonPathAnalyzerPerSL() {
  if (debug_)
    cout << "MuonPathAnalyzer: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void MuonPathAnalyzerPerSL::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    cout << "MuonPathAnalyzerPerSL::initialiase" << endl;

  const MuonGeometryRecord &geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void MuonPathAnalyzerPerSL::run(edm::Event &iEvent,
                                const edm::EventSetup &iEventSetup,
                                std::vector<MuonPath *> &muonpaths,
                                std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    cout << "MuonPathAnalyzerPerSL: run" << endl;

  // fit per SL (need to allow for multiple outputs for a single mpath)
  for (auto muonpath = muonpaths.begin(); muonpath != muonpaths.end(); ++muonpath) {
    analyze(*muonpath, metaPrimitives);
  }
}

void MuonPathAnalyzerPerSL::finish() {
  if (debug_)
    cout << "MuonPathAnalyzer: finish" << endl;
};

const int MuonPathAnalyzerPerSL::LAYER_ARRANGEMENTS_[4][3] = {
    {0, 1, 2},
    {1, 2, 3},  // Grupos consecutivos
    {0, 1, 3},
    {0, 2, 3}  // Grupos salteados
};

//------------------------------------------------------------------
//--- Métodos privados
//------------------------------------------------------------------

void MuonPathAnalyzerPerSL::analyze(MuonPath *inMPath, std::vector<metaPrimitive> &metaPrimitives) {
  if (debug_)
    std::cout << "DTp2:analyze \t\t\t\t starts" << std::endl;

  // LOCATE MPATH
  int selected_Id = 0;
  if (inMPath->primitive(0)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(0)->cameraId();
  else if (inMPath->primitive(1)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(1)->cameraId();
  else if (inMPath->primitive(2)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(2)->cameraId();
  else if (inMPath->primitive(3)->tdcTimeStamp() != -1)
    selected_Id = inMPath->primitive(3)->cameraId();

  DTLayerId thisLId(selected_Id);
  if (debug_)
    std::cout << "Building up MuonPathSLId from rawId in the Primitive" << std::endl;
  DTSuperLayerId MuonPathSLId(thisLId.wheel(), thisLId.station(), thisLId.sector(), thisLId.superLayer());
  if (debug_)
    std::cout << "The MuonPathSLId is" << MuonPathSLId << std::endl;

  if (debug_)
    std::cout << "DTp2:analyze \t\t\t\t In analyze function checking if inMPath->isAnalyzable() "
              << inMPath->isAnalyzable() << std::endl;

  if (chosen_sl_ < 4 && thisLId.superLayer() != chosen_sl_)
    return;  // avoid running when mpath not in chosen SL (for 1SL fitting)

  // Clonamos el objeto analizado.
  MuonPath *mPath = new MuonPath(inMPath);

  if (mPath->isAnalyzable()) {
    if (debug_)
      std::cout << "DTp2:analyze \t\t\t\t\t yes it is analyzable " << mPath->isAnalyzable() << std::endl;
    setCellLayout(mPath->cellLayout());
    evaluatePathQuality(mPath);
  } else {
    if (debug_)
      std::cout << "DTp2:analyze \t\t\t\t\t no it is NOT analyzable " << mPath->isAnalyzable() << std::endl;
    return;
  }

  int wi[8], tdc[8], lat[8];
  DTPrimitive Prim0(mPath->primitive(0));
  wi[0] = Prim0.channelId();
  tdc[0] = Prim0.tdcTimeStamp();
  DTPrimitive Prim1(mPath->primitive(1));
  wi[1] = Prim1.channelId();
  tdc[1] = Prim1.tdcTimeStamp();
  DTPrimitive Prim2(mPath->primitive(2));
  wi[2] = Prim2.channelId();
  tdc[2] = Prim2.tdcTimeStamp();
  DTPrimitive Prim3(mPath->primitive(3));
  wi[3] = Prim3.channelId();
  tdc[3] = Prim3.tdcTimeStamp();
  for (int i = 4; i < 8; i++) {
    wi[i] = -1;
    tdc[i] = -1;
    lat[i] = -1;
  }

  DTWireId wireId(MuonPathSLId, 2, 1);

  if (debug_)
    std::cout << "DTp2:analyze \t\t\t\t checking if it passes the min quality cut " << mPath->quality() << ">"
              << minQuality_ << std::endl;
  if (mPath->quality() >= minQuality_) {
    if (debug_)
      std::cout << "DTp2:analyze \t\t\t\t min quality achievedCalidad: " << mPath->quality() << std::endl;
    for (int i = 0; i <= 3; i++) {
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t  Capa: " << mPath->primitive(i)->layerId()
                  << " Canal: " << mPath->primitive(i)->channelId()
                  << " TDCTime: " << mPath->primitive(i)->tdcTimeStamp() << std::endl;
    }
    if (debug_)
      std::cout << "DTp2:analyze \t\t\t\t Starting lateralities loop, totalNumValLateralities: "
                << totalNumValLateralities_ << std::endl;

    double best_chi2 = 99999.;
    double chi2_jm_tanPhi = 999;
    double chi2_jm_x = -1;
    double chi2_jm_t0 = -1;
    double chi2_phi = -1;
    double chi2_phiB = -1;
    double chi2_chi2 = -1;
    int chi2_quality = -1;
    int bestLat[8];
    for (int i = 0; i < 8; i++) {
      bestLat[i] = -1;
    }

    for (int i = 0; i < totalNumValLateralities_; i++) {  //here
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << std::endl;
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " checking quality:" << std::endl;
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " checking mPath Quality=" << mPath->quality()
                  << std::endl;
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " latQuality_[i].val=" << latQuality_[i].valid
                  << std::endl;
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " before if:" << std::endl;

      if (latQuality_[i].valid and
          (((mPath->quality() == HIGHQ or mPath->quality() == HIGHQGHOST) and latQuality_[i].quality == HIGHQ) or
           ((mPath->quality() == LOWQ or mPath->quality() == LOWQGHOST) and latQuality_[i].quality == LOWQ))) {
        if (debug_)
          std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " inside if" << std::endl;
        mPath->setBxTimeValue(latQuality_[i].bxValue);
        if (debug_)
          std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " settingLateralCombination" << std::endl;
        mPath->setLateralComb(lateralities_[i]);
        if (debug_)
          std::cout << "DTp2:analyze \t\t\t\t\t laterality #- " << i << " done settingLateralCombination" << std::endl;

        // Clonamos el objeto analizado.
        MuonPath *mpAux = new MuonPath(mPath);
        lat[0] = mpAux->lateralComb()[0];
        lat[1] = mpAux->lateralComb()[1];
        lat[2] = mpAux->lateralComb()[2];
        lat[3] = mpAux->lateralComb()[3];

        int wiOk[4], tdcOk[4], latOk[4];
        for (int lay = 0; lay < 4; lay++) {
          if (latQuality_[i].invalidateHitIdx == lay) {
            wiOk[lay] = -1;
            tdcOk[lay] = -1;
            latOk[lay] = -1;
          } else {
            wiOk[lay] = wi[lay];
            tdcOk[lay] = tdc[lay];
            latOk[lay] = lat[lay];
          }
        }

        int idxHitNotValid = latQuality_[i].invalidateHitIdx;
        if (idxHitNotValid >= 0) {
          delete mpAux->primitive(idxHitNotValid);
          mpAux->setPrimitive(std::move(new DTPrimitive()), idxHitNotValid);
        }

        if (debug_)
          std::cout << "DTp2:analyze \t\t\t\t\t  calculating parameters " << std::endl;
        calculatePathParameters(mpAux);
        /* 
		 * Si, tras calcular los parámetros, y si se trata de un segmento
		 * con 4 hits, el chi2 resultante es superior al umbral programado,
		 * lo eliminamos y no se envía al exterior.
		 * Y pasamos al siguiente elemento.
		 */
        if ((mpAux->quality() == HIGHQ or mpAux->quality() == HIGHQGHOST) &&
            mpAux->chiSquare() > chiSquareThreshold_) {  //check this if!!!
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  HIGHQ or HIGHQGHOST but min chi2 or Q test not satisfied "
                      << std::endl;
        } else {
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  inside else, returning values: " << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  BX Time = " << mpAux->bxTimeValue() << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  BX Id   = " << mpAux->bxNumId() << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  XCoor   = " << mpAux->horizPos() << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  tan(Phi)= " << mpAux->tanPhi() << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  chi2= " << mpAux->chiSquare() << std::endl;
          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t  lateralities = "
                      << " " << mpAux->lateralComb()[0] << " " << mpAux->lateralComb()[1] << " "
                      << mpAux->lateralComb()[2] << " " << mpAux->lateralComb()[3] << std::endl;

          DTChamberId ChId(MuonPathSLId.wheel(), MuonPathSLId.station(), MuonPathSLId.sector());

          double jm_tanPhi = -1. * mpAux->tanPhi();  //testing with this line
          //printf ("jm_tanPhi=%f jm_tanPhi / tanPsi_precision=%f floor(jm_tanPhi / tanPsi_precision=%f \n", jm_tanPhi, jm_tanPhi / tanPsi_precision, floor(jm_tanPhi / tanPsi_precision));
          if (use_LSB_)
            jm_tanPhi = floor(jm_tanPhi / tanPsi_precision_) * tanPsi_precision_;
          //if (use_LSB) jm_tanPhi = floor(jm_tanPhi / tanPsi_precision)  * tanPsi_precision;
          double jm_x =
              (((double)mpAux->horizPos()) / 10.) + x_precision_ * (round(shiftinfo_[wireId.rawId()] / x_precision_));
          //cout << (((double) mpAux->horizPos())/10.) <<" "<<((double)shiftinfo[wireId.rawId()]) << " " << jm_x / x_precision_ << " " << round(jm_x / x_precision_) << endl;
          if (use_LSB_)
            jm_x = ((double)round(((double)jm_x) / x_precision_)) * x_precision_;
          //if (use_LSB) jm_x = floor(jm_x / x_precision_)  * x_precision_;
          //changing to chamber frame or reference:
          double jm_t0 = mpAux->bxTimeValue();
          int quality = mpAux->quality();

          //computing phi and phiB
          double z = 0;
          double z1 = 11.75;
          double z3 = -1. * z1;
          if (ChId.station() == 3 or ChId.station() == 4) {
            z1 = 9.95;
            z3 = -13.55;
          }
          if (MuonPathSLId.superLayer() == 1)
            z = z1;
          if (MuonPathSLId.superLayer() == 3)
            z = z3;

          GlobalPoint jm_x_cmssw_global = dtGeo_->chamber(ChId)->toGlobal(LocalPoint(jm_x, 0., z));
          int thisec = MuonPathSLId.sector();
          if (thisec == 13)
            thisec = 4;
          if (thisec == 14)
            thisec = 10;
          double phi = jm_x_cmssw_global.phi() - 0.5235988 * (thisec - 1);
          double psi = atan(jm_tanPhi);
          double phiB = hasPosRF(MuonPathSLId.wheel(), MuonPathSLId.sector()) ? psi - phi : -psi - phi;
          double chi2 = mpAux->chiSquare() * 0.01;  //in cmssw we need cm, 1 cm^2 = 100 mm^2

          if (debug_)
            std::cout << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaPrimitive at x=" << jm_x
                      << " tanPhi:" << jm_tanPhi << " t0:" << jm_t0 << std::endl;

          if (mpAux->quality() == HIGHQ or
              mpAux->quality() == HIGHQGHOST) {  //keep only the values with the best chi2 among lateralities
            if ((chi2 < best_chi2) && (fabs(jm_tanPhi) <= tanPhiTh_)) {
              chi2_jm_tanPhi = jm_tanPhi;
              chi2_jm_x = (mpAux->horizPos() / 10.) + shiftinfo_[wireId.rawId()];
              chi2_jm_t0 = mpAux->bxTimeValue();
              chi2_phi = phi;
              chi2_phiB = phiB;
              chi2_chi2 = chi2;
              best_chi2 = chi2;
              chi2_quality = mpAux->quality();
              for (int i = 0; i < 4; i++) {
                bestLat[i] = lat[i];
              }
            }
          } else if (fabs(jm_tanPhi) <=
                     tanPhiTh_) {  //write the metaprimitive in case no HIGHQ or HIGHQGHOST and tanPhi range
            if (debug_)
              std::cout << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back metaprimitive no HIGHQ or HIGHQGHOST"
                        << std::endl;
            metaPrimitives.push_back(metaPrimitive({MuonPathSLId.rawId(),
                                                    jm_t0,
                                                    jm_x,
                                                    jm_tanPhi,
                                                    phi,
                                                    phiB,
                                                    chi2,
                                                    quality,
                                                    wiOk[0],
                                                    tdcOk[0],
                                                    latOk[0],
                                                    wiOk[1],
                                                    tdcOk[1],
                                                    latOk[1],
                                                    wiOk[2],
                                                    tdcOk[2],
                                                    latOk[2],
                                                    wiOk[3],
                                                    tdcOk[3],
                                                    latOk[3],
                                                    wi[4],
                                                    tdc[4],
                                                    lat[4],
                                                    wi[5],
                                                    tdc[5],
                                                    lat[5],
                                                    wi[6],
                                                    tdc[6],
                                                    lat[6],
                                                    wi[7],
                                                    tdc[7],
                                                    lat[7],
                                                    -1}));
            if (debug_)
              std::cout << "DTp2:analyze \t\t\t\t\t\t\t\t  done pushing back metaprimitive no HIGHQ or HIGHQGHOST"
                        << std::endl;
          }
        }
        delete mpAux;
      } else {
        if (debug_)
          std::cout << "DTp2:analyze \t\t\t\t\t\t\t\t  latQuality_[i].valid and (((mPath->quality()==HIGHQ or "
                       "mPath->quality()==HIGHQGHOST) and latQuality_[i].quality==HIGHQ) or  ((mPath->quality() "
                       "== LOWQ or mPath->quality()==LOWQGHOST) and latQuality_[i].quality==LOWQ)) not passed"
                    << std::endl;
      }
    }
    if (chi2_jm_tanPhi != 999 and fabs(chi2_jm_tanPhi) < tanPhiTh_) {  //
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t\t\t\t  pushing back best chi2 metaPrimitive" << std::endl;
      metaPrimitives.push_back(metaPrimitive({MuonPathSLId.rawId(),
                                              chi2_jm_t0,
                                              chi2_jm_x,
                                              chi2_jm_tanPhi,
                                              chi2_phi,
                                              chi2_phiB,
                                              chi2_chi2,
                                              chi2_quality,
                                              wi[0],
                                              tdc[0],
                                              bestLat[0],
                                              wi[1],
                                              tdc[1],
                                              bestLat[1],
                                              wi[2],
                                              tdc[2],
                                              bestLat[2],
                                              wi[3],
                                              tdc[3],
                                              bestLat[3],
                                              wi[4],
                                              tdc[4],
                                              bestLat[4],
                                              wi[5],
                                              tdc[5],
                                              bestLat[5],
                                              wi[6],
                                              tdc[6],
                                              bestLat[6],
                                              wi[7],
                                              tdc[7],
                                              bestLat[7],
                                              -1}));
    }
  }
  delete mPath;
  if (debug_)
    std::cout << "DTp2:analyze \t\t\t\t finishes" << std::endl;
}

void MuonPathAnalyzerPerSL::setCellLayout(const int layout[4]) {
  memcpy(cellLayout_, layout, 4 * sizeof(int));
  //celllayout[0]=layout[0];
  //celllayout[1]=layout[1];
  //celllayout[2]=layout[2];
  //celllayout[3]=layout[3];

  buildLateralities();
}

/**
 * Para una combinación de 4 celdas dada (las que se incluyen en el analizador,
 * una por capa), construye de forma automática todas las posibles
 * combinaciones de lateralidad (LLLL, LLRL,...) que son compatibles con una
 * trayectoria recta. Es decir, la partícula no hace un zig-zag entre los hilos
 * de diferentes celdas, al pasar de una a otra.
 */
void MuonPathAnalyzerPerSL::buildLateralities(void) {
  LATERAL_CASES(*validCase)[4], sideComb[4];

  totalNumValLateralities_ = 0;
  /* Generamos todas las posibles combinaciones de lateralidad para el grupo
       de celdas que forman parte del analizador */
  for (int lowLay = LEFT; lowLay <= RIGHT; lowLay++)
    for (int midLowLay = LEFT; midLowLay <= RIGHT; midLowLay++)
      for (int midHigLay = LEFT; midHigLay <= RIGHT; midHigLay++)
        for (int higLay = LEFT; higLay <= RIGHT; higLay++) {
          sideComb[0] = static_cast<LATERAL_CASES>(lowLay);
          sideComb[1] = static_cast<LATERAL_CASES>(midLowLay);
          sideComb[2] = static_cast<LATERAL_CASES>(midHigLay);
          sideComb[3] = static_cast<LATERAL_CASES>(higLay);

          /* Si una combinación de lateralidades es válida, la almacenamos */
          if (isStraightPath(sideComb)) {
            validCase = lateralities_ + totalNumValLateralities_;
            memcpy(validCase, sideComb, 4 * sizeof(LATERAL_CASES));

            latQuality_[totalNumValLateralities_].valid = false;
            latQuality_[totalNumValLateralities_].bxValue = 0;
            latQuality_[totalNumValLateralities_].quality = NOPATH;
            latQuality_[totalNumValLateralities_].invalidateHitIdx = -1;

            totalNumValLateralities_++;
          }
        }
}

/**
 * Para automatizar la generación de trayectorias compatibles con las posibles
 * combinaciones de lateralidad, este método decide si una cierta combinación
 * de lateralidad, involucrando 4 celdas de las que conforman el MuonPathAnalyzerPerSL,
 * forma una traza recta o no. En caso negativo, la combinación de lateralidad
 * es descartada y no se analiza.
 * En el caso de implementación en FPGA, puesto que el diseño intentará
 * paralelizar al máximo la lógica combinacional, el equivalente a este método
 * debería ser un "generate" que expanda las posibles combinaciones de
 * lateralidad de celdas compatibles con el análisis.
 *
 * El métoda da por válida una trayectoria (es recta) si algo parecido al
 * cambio en la pendiente de la trayectoria, al cambiar de par de celdas
 * consecutivas, no es mayor de 1 en unidades arbitrarias de semi-longitudes
 * de celda para la dimensión horizontal, y alturas de celda para la vertical.
 */
bool MuonPathAnalyzerPerSL::isStraightPath(LATERAL_CASES sideComb[4]) {
  return true;  //trying with all lateralities to be confirmed

  int i, ajustedLayout[4], pairDiff[3], desfase[3];

  /* Sumamos el valor de lateralidad (LEFT = 0, RIGHT = 1) al desfase
       horizontal (respecto de la celda base) para cada celda en cuestion */
  for (i = 0; i <= 3; i++)
    ajustedLayout[i] = cellLayout_[i] + sideComb[i];
  /* Variación del desfase por pares de celdas consecutivas */
  for (i = 0; i <= 2; i++)
    pairDiff[i] = ajustedLayout[i + 1] - ajustedLayout[i];
  /* Variación de los desfases entre todas las combinaciones de pares */
  for (i = 0; i <= 1; i++)
    desfase[i] = abs(pairDiff[i + 1] - pairDiff[i]);
  desfase[2] = abs(pairDiff[2] - pairDiff[0]);
  /* Si algún desfase es mayor de 2 entonces la trayectoria no es recta */
  bool resultado = (desfase[0] > 1 or desfase[1] > 1 or desfase[2] > 1);

  return (!resultado);
}

/**
 * Recorre las calidades calculadas para todas las combinaciones de lateralidad
 * válidas, para determinar la calidad final asignada al "MuonPath" con el que
 * se está trabajando.
 */
void MuonPathAnalyzerPerSL::evaluatePathQuality(MuonPath *mPath) {
  // here
  int totalHighQ = 0, totalLowQ = 0;

  if (debug_)
    std::cout << "DTp2:evaluatePathQuality \t\t\t\t\t En evaluatePathQuality Evaluando PathQ. Celda base: "
              << mPath->baseChannelId() << std::endl;
  if (debug_)
    std::cout << "DTp2:evaluatePathQuality \t\t\t\t\t Total lateralidades: " << totalNumValLateralities_ << std::endl;

  // Por defecto.
  mPath->setQuality(NOPATH);

  /* Ensayamos los diferentes grupos de lateralidad válidos que constituyen
       las posibles trayectorias del muón por el grupo de 4 celdas.
       Posiblemente esto se tenga que optimizar de manera que, si en cuanto se
       encuentre una traza 'HIGHQ' ya no se continue evaluando mas combinaciones
       de lateralidad, pero hay que tener en cuenta los fantasmas (rectas
       paralelas) en de alta calidad que se pueden dar en los extremos del BTI.
       Posiblemente en la FPGA, si esto se paraleliza, no sea necesaria tal
       optimización */
  for (int latIdx = 0; latIdx < totalNumValLateralities_; latIdx++) {
    if (debug_)
      std::cout << "DTp2:evaluatePathQuality \t\t\t\t\t Analizando combinacion de lateralidad: "
                << lateralities_[latIdx][0] << " " << lateralities_[latIdx][1] << " " << lateralities_[latIdx][2] << " "
                << lateralities_[latIdx][3] << std::endl;

    evaluateLateralQuality(latIdx, mPath, &(latQuality_[latIdx]));

    if (latQuality_[latIdx].quality == HIGHQ) {
      totalHighQ++;
      if (debug_)
        std::cout << "DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad HIGHQ" << std::endl;
    }
    if (latQuality_[latIdx].quality == LOWQ) {
      totalLowQ++;
      if (debug_)
        std::cout << "DTp2:evaluatePathQuality \t\t\t\t\t\t Lateralidad LOWQ" << std::endl;
    }
  }
  /*
     * Establecimiento de la calidad.
     */
  if (totalHighQ == 1) {
    mPath->setQuality(HIGHQ);
  } else if (totalHighQ > 1) {
    mPath->setQuality(HIGHQGHOST);
  } else if (totalLowQ == 1) {
    mPath->setQuality(LOWQ);
  } else if (totalLowQ > 1) {
    mPath->setQuality(LOWQGHOST);
  }
}

void MuonPathAnalyzerPerSL::evaluateLateralQuality(int latIdx, MuonPath *mPath, LATQ_TYPE *latQuality) {
  int layerGroup[3];
  LATERAL_CASES sideComb[3];
  PARTIAL_LATQ_TYPE latQResult[4] = {{false, 0}, {false, 0}, {false, 0}, {false, 0}};

  // Default values.
  latQuality->valid = false;
  latQuality->bxValue = 0;
  latQuality->quality = NOPATH;
  latQuality->invalidateHitIdx = -1;

  /* En el caso que, para una combinación de lateralidad dada, las 2
       combinaciones consecutivas de 3 capas ({0, 1, 2}, {1, 2, 3}) fueran
       traza válida, habríamos encontrado una traza correcta de alta calidad,
       por lo que sería innecesario comprobar las otras 2 combinaciones
       restantes.
       Ahora bien, para reproducir el comportamiento paralelo de la FPGA en el
       que el análisis se va a evaluar simultáneamente en todas ellas,
       construimos un código que analiza las 4 combinaciones, junto con una
       lógica adicional para discriminar la calidad final de la traza */
  for (int i = 0; i <= 3; i++) {
    memcpy(layerGroup, LAYER_ARRANGEMENTS_[i], 3 * sizeof(int));

    // Seleccionamos la combinación de lateralidad para cada celda.
    for (int j = 0; j < 3; j++)
      sideComb[j] = lateralities_[latIdx][layerGroup[j]];

    validate(sideComb, layerGroup, mPath, &(latQResult[i]));
  }
  /*
      Imponemos la condición, para una combinación de lateralidad completa, que
      todas las lateralidades parciales válidas arrojen el mismo valor de BX
      (dentro de un margen) para así dar una traza consistente.
      En caso contrario esa combinación se descarta.
    */
  if (!sameBXValue(latQResult)) {
    // Se guardan en los default values inciales.
    if (debug_)
      std::cout << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. Tolerancia de BX excedida"
                << std::endl;
    return;
  }

  // Dos trazas complementarias válidas => Traza de muón completa.
  if ((latQResult[0].latQValid && latQResult[1].latQValid) or (latQResult[0].latQValid && latQResult[2].latQValid) or
      (latQResult[0].latQValid && latQResult[3].latQValid) or (latQResult[1].latQValid && latQResult[2].latQValid) or
      (latQResult[1].latQValid && latQResult[3].latQValid) or (latQResult[2].latQValid && latQResult[3].latQValid)) {
    latQuality->valid = true;
    //     latQuality->bxValue = latQResult[0].bxValue;
    /*
	     * Se hace necesario el contador de casos "numValid", en vez de promediar
	     * los 4 valores dividiendo entre 4, puesto que los casos de combinaciones
	     * de 4 hits buenos que se ajusten a una combinación como por ejemplo:
	     * L/R/L/L, dan lugar a que en los subsegmentos 0, y 1 (consecutivos) se
	     * pueda aplicar mean-timer, mientras que en el segmento 3 (en el ejemplo
	     * capas: 0,2,3, y combinación L/L/L) no se podría aplicar, dando un
	     * valor parcial de BX = 0.
	     */
    if (debug_)
      std::cout << "DTp2:analyze \t\t\t\t\t\t Valid BXs" << std::endl;
    long int sumBX = 0, numValid = 0;
    for (int i = 0; i <= 3; i++) {
      if (debug_)
        std::cout << "DTp2:analyze \t\t\t\t\t\t "
                  << "[" << latQResult[i].bxValue << "," << latQResult[i].latQValid << "]" << endl;
      if (latQResult[i].latQValid) {
        sumBX += latQResult[i].bxValue;
        //		    cout <<  " BX:" << latQResult[i].bxValue << " tdc" << mPath->primitive(i)->tdcTimeStamp() << endl;
        numValid++;
      }
    }

    if (numValid == 1)
      latQuality->bxValue = sumBX;
    else if (numValid == 2)
      latQuality->bxValue = (sumBX * (16384)) / std::pow(2, 15);
    else if (numValid == 3)
      latQuality->bxValue = (sumBX * (10923)) / std::pow(2, 15);
    //else if (numValid == 3) latQuality->bxValue = (sumBX * (10922) ) / std::pow(2,15);
    else if (numValid == 4)
      latQuality->bxValue = (sumBX * (8192)) / std::pow(2, 15);

    //	    cout << "MEDIA BX:" << latQuality->bxValue << " Validos:" << numValid << endl;

    latQuality->quality = HIGHQ;

    if (debug_)
      std::cout << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. HIGHQ." << std::endl;
  }
  // Sólo una traza disjunta válida => Traza de muón incompleta pero válida.
  else {
    if (latQResult[0].latQValid or latQResult[1].latQValid or latQResult[2].latQValid or latQResult[3].latQValid) {
      latQuality->valid = true;
      latQuality->quality = LOWQ;
      for (int i = 0; i < 4; i++)
        if (latQResult[i].latQValid) {
          latQuality->bxValue = latQResult[i].bxValue;
          //latQuality->bxValue = floor ( latQResult[i].bxValue * (pow(2,15)-1) / (pow(2,15)) );
          /*
			 * En los casos que haya una combinación de 4 hits válidos pero
			 * sólo 3 de ellos formen traza (calidad 2), esto permite detectar
			 * la layer con el hit que no encaja en la recta, y así poder
			 * invalidarlo, cambiando su valor por "-1" como si de una mezcla
			 * de 3 hits pura se tratara.
			 * Esto es útil para los filtros posteriores.
			 */
          latQuality->invalidateHitIdx = omittedHit(i);
          break;
        }

      if (debug_)
        std::cout << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad ACEPTADA. LOWQ." << std::endl;
    } else {
      if (debug_)
        std::cout << "DTp2:evaluateLateralQuality \t\t\t\t\t Lateralidad DESCARTADA. NOPATH." << std::endl;
    }
  }
}

/**
 * Valida, para una combinación de capas (3), celdas y lateralidad, si los
 * valores temporales cumplen el criterio de mean-timer.
 * En vez de comparar con un 0 estricto, que es el resultado aritmético de las
 * ecuaciones usadas de base, se incluye en la clase un valor de tolerancia
 * que por defecto vale cero, pero que se puede ajustar a un valor más
 * adecuado
 *
 * En esta primera versión de la clase, el código de generación de ecuaciones
 * se incluye en esta función, lo que es ineficiente porque obliga a calcular
 * un montón de constantes, fijas para cada combinación de celdas, que
 * tendrían que evaluarse una sóla vez en el constructor de la clase.
 * Esta disposición en el constructor estaría más proxima a la realización que
 * se tiene que llevar a término en la FPGA (en tiempo de síntesis).
 * De momento se deja aquí porque así se entiende la lógica mejor, al estar
 * descrita de manera lineal en un sólo método.
 */
void MuonPathAnalyzerPerSL::validate(LATERAL_CASES sideComb[3],
                                     int layerIndex[3],
                                     MuonPath *mPath,
                                     PARTIAL_LATQ_TYPE *latq) {
  // Valor por defecto.
  latq->bxValue = 0;
  latq->latQValid = false;

  if (debug_)
    std::cout << "DTp2:validate \t\t\t\t\t\t\t In validate Iniciando validacion de MuonPath para capas: "
              << layerIndex[0] << "/" << layerIndex[1] << "/" << layerIndex[2] << std::endl;

  if (debug_)
    std::cout << "DTp2:validate \t\t\t\t\t\t\t Lateralidades parciales: " << sideComb[0] << "/" << sideComb[1] << "/"
              << sideComb[2] << std::endl;

  /* Primero evaluamos si, para la combinación concreta de celdas en curso, el
       número de celdas con dato válido es 3. Si no es así, sobre esa
       combinación no se puede aplicar el mean-timer y devolvemos "false" */
  int validCells = 0;
  for (int j = 0; j < 3; j++)
    if (mPath->primitive(layerIndex[j])->isValidTime())
      validCells++;

  if (validCells != 3) {
    if (debug_)
      std::cout << "DTp2:validate \t\t\t\t\t\t\t No hay 3 celdas validas." << std::endl;
    return;
  }

  if (debug_)
    std::cout << "DTp2:validate \t\t\t\t\t\t\t Valores de TDC: " << mPath->primitive(layerIndex[0])->tdcTimeStamp()
              << "/" << mPath->primitive(layerIndex[1])->tdcTimeStamp() << "/"
              << mPath->primitive(layerIndex[2])->tdcTimeStamp() << "." << std::endl;

  if (debug_)
    std::cout << "DTp2:validate \t\t\t\t\t\t\t Valid TIMES: " << mPath->primitive(layerIndex[0])->isValidTime() << "/"
              << mPath->primitive(layerIndex[1])->isValidTime() << "/" << mPath->primitive(layerIndex[2])->isValidTime()
              << "." << std::endl;

  /* Distancias verticales entre capas inferior/media y media/superior */
  int dVertMI = layerIndex[1] - layerIndex[0];
  int dVertSM = layerIndex[2] - layerIndex[1];

  /* Distancias horizontales entre capas inferior/media y media/superior */
  int dHorzMI = cellLayout_[layerIndex[1]] - cellLayout_[layerIndex[0]];
  int dHorzSM = cellLayout_[layerIndex[2]] - cellLayout_[layerIndex[1]];

  /* Índices de pares de capas sobre las que se está actuando
       SM => Superior + Intermedia
       MI => Intermedia + Inferior
       Jugamos con los punteros para simplificar el código */
  int *layPairSM = &layerIndex[1];
  int *layPairMI = &layerIndex[0];

  /* Pares de combinaciones de celdas para composición de ecuación. Sigue la
       misma nomenclatura que el caso anterior */
  LATERAL_CASES smSides[2], miSides[2];

  /* Teniendo en cuenta que en el índice 0 de "sideComb" se almacena la
       lateralidad de la celda inferior, jugando con aritmética de punteros
       extraemos las combinaciones de lateralidad para los pares SM y MI */

  memcpy(smSides, &sideComb[1], 2 * sizeof(LATERAL_CASES));

  memcpy(miSides, &sideComb[0], 2 * sizeof(LATERAL_CASES));

  long int bxValue = 0;
  //double bxValue = 0;
  int coefsAB[2] = {0, 0}, coefsCD[2] = {0, 0};
  /* It's neccesary to be careful with that pointer's indirection. We need to
       retrieve the lateral coeficientes (+-1) from the lower/middle and
       middle/upper cell's lateral combinations. They are needed to evaluate the
       existance of a possible BX value, following it's calculation equation */
  lateralCoeficients(miSides, coefsAB);
  lateralCoeficients(smSides, coefsCD);

  /* Cada para de sumas de los 'coefsCD' y 'coefsAB' dan siempre como resultado
       0, +-2.

       A su vez, y pese a que las ecuaciones se han construido de forma genérica
       para cualquier combinación de celdas de la cámara, los valores de 'dVertMI' y
       'dVertSM' toman valores 1 o 2 puesto que los pares de celdas con los que se
       opera en realidad, o bien están contiguos, o bien sólo están separadas por
       una fila de celdas intermedia. Esto es debido a cómo se han combinado los
       grupos de celdas, para aplicar el mean-timer, en 'LAYER_ARRANGEMENTS'.

       El resultado final es que 'denominator' es siempre un valor o nulo, o
       múltiplo de 2 */
  int denominator = dVertMI * (coefsCD[1] + coefsCD[0]) - dVertSM * (coefsAB[1] + coefsAB[0]);

  if (denominator == 0) {
    if (debug_)
      std::cout << "DTp2:validate \t\t\t\t\t\t\t Imposible calcular BX. Denominador para BX = 0." << std::endl;
    return;
  }

  /* Esta ecuación ha de ser optimizada, especialmente en su implementación
       en FPGA. El 'denominator' toma siempre valores múltiplo de 2 o nulo, por lo
       habría que evitar el cociente y reemplazarlo por desplazamientos de bits */
  /*bxValue = (
	       dVertMI*(dHorzSM*MAXDRIFT + eqMainBXTerm(smSides, layPairSM, mPath)) -
	       dVertSM*(dHorzMI*MAXDRIFT + eqMainBXTerm(miSides, layPairMI, mPath))
	       ) / denominator;
   */ //MODIFIED BY ALVARO
  //long int sumA = (long int) floor(MAXDRIFT * (dVertMI*dHorzSM - dVertSM*dHorzMI));
  long int sumA = (long int)floor(MAXDRIFT * (dVertMI * dHorzSM - dVertSM * dHorzMI));
  //cout << "sumA " << sumA << endl;
  long int numerator =
      (sumA + dVertMI * eqMainBXTerm(smSides, layPairSM, mPath) - dVertSM * eqMainBXTerm(miSides, layPairMI, mPath));
  //long int numerator = (  (sumA)>> 2 + dVertMI*eqMainBXTerm(smSides, layPairSM, mPath) - dVertSM*eqMainBXTerm(miSides, layPairMI, mPath));

  //long int numerator = ( dVertMI*(dHorzSM*MAXDRIFT + eqMainBXTerm(smSides, layPairSM, mPath)) -
  //	       dVertSM*(dHorzMI*MAXDRIFT + eqMainBXTerm(miSides, layPairMI, mPath)));
  //if (denominator == -6)      bxValue = (numerator * (-5461 ) ) / std::pow(2,15);
  //else if (denominator == -4) bxValue = (numerator * (-8192 ) ) / std::pow(2,15);
  //else if (denominator == -2) bxValue = (numerator * (-16384) ) / std::pow(2,15);
  //else if (denominator == 2)  bxValue = (numerator * ( 16384) ) / std::pow(2,15);
  //else if (denominator == 4)  bxValue = (numerator * ( 8192 ) ) / std::pow(2,15);
  //else if (denominator == 6)  bxValue = (numerator * ( 5461 ) ) / std::pow(2,15);
  if (denominator == -6)
    bxValue = (numerator * (-43691)) / std::pow(2, 18);
  else if (denominator == -4)
    bxValue = (numerator * (-65536)) / std::pow(2, 18);
  else if (denominator == -2)
    bxValue = (numerator * (-131072)) / std::pow(2, 18);
  else if (denominator == 2)
    bxValue = (numerator * (131072)) / std::pow(2, 18);
  else if (denominator == 4)
    bxValue = (numerator * (65536)) / std::pow(2, 18);
  else if (denominator == 6)
    bxValue = (numerator * (43691)) / std::pow(2, 18);
  else
    cout << "Distinto!" << endl;
  //cout << numerator * 5461 << " " << std::pow(2,15) << " "; printf("%f\n",bxValue);
  //bxValue = floor (bxValue);
  //cout << "bxValue " << bxValue << endl;
  //cout << bxValue << " numerator: " << numerator << " denominator: " << denominator << endl;
  if (bxValue < 0) {
    if (debug_)
      std::cout << "DTp2:validate \t\t\t\t\t\t\t Combinacion no valida. BX Negativo." << std::endl;
    return;
  }
  /*std::cout<<"Valores de TDC: "
                       <<mPath->primitive(layerIndex[0])->tdcTimeStamp()<<"/"
                       <<mPath->primitive(layerIndex[1])->tdcTimeStamp()<<"/"
                       <<mPath->primitive(layerIndex[2])->tdcTimeStamp()<<"."
                      << " BXvalue " << bxValue << std::endl; */

  // Redondeo del valor del tiempo de BX al nanosegundo
  //if ( (bxValue - int(bxValue)) >= 0.5 ) bxValue = float(int(bxValue + 1));
  //else bxValue = float(int(bxValue));

  /* Ciertos valores del tiempo de BX, siendo positivos pero objetivamente no
       válidos, pueden dar lugar a que el discriminador de traza asociado de un
       valor aparentemente válido (menor que la tolerancia y típicamente 0). Eso es
       debido a que el valor de tiempo de BX es mayor que algunos de los tiempos
       de TDC almacenados en alguna de las respectivas 'DTPrimitives', lo que da
       lugar a que, cuando se establece el valore de BX para el 'MuonPath', se
       obtengan valores de tiempo de deriva (*NO* tiempo de TDC) en la 'DTPrimitive'
       nulos, o inconsistentes, a causa de la resta entre enteros.

       Así pues, se impone como criterio de validez adicional que el valor de tiempo
       de BX (bxValue) sea siempre superior a cualesquiera valores de tiempo de TDC
       almacenados en las 'DTPrimitives' que forman el 'MuonPath' que se está
       analizando.
       En caso contrario, se descarta como inválido */

  for (int i = 0; i < 3; i++)
    if (mPath->primitive(layerIndex[i])->isValidTime()) {
      int diffTime = mPath->primitive(layerIndex[i])->tdcTimeStampNoOffset() - bxValue;
      //cout << bxValue  <<  " " << mPath->primitive(layerIndex[i])->tdcTimeStampNoOffset() << endl;

      if (diffTime <= 0 or diffTime > round(MAXDRIFT)) {
        //if (diffTime < 0 or diffTime > MAXDRIFT) {
        if (debug_)
          std::cout << "DTp2:validate \t\t\t\t\t\t\t Valor de BX inválido. Al menos un tiempo de TDC sin sentido"
                    << std::endl;
        return;
      }
    }
  if (debug_)
    std::cout << "DTp2:validate \t\t\t\t\t\t\t Valor de BX: " << bxValue << std::endl;

  /* Si se llega a este punto, el valor de BX y la lateralidad parcial se dan
     * por válidas.
     */
  latq->bxValue = bxValue;
  latq->latQValid = true;
}  //finish validate

/**
 * Evalúa la suma característica de cada par de celdas, según la lateralidad
 * de la trayectoria.
 * El orden de los índices de capa es crítico:
 *    layerIdx[0] -> Capa más baja,
 *    layerIdx[1] -> Capa más alta
 */
int MuonPathAnalyzerPerSL::eqMainBXTerm(LATERAL_CASES sideComb[2], int layerIdx[2], MuonPath *mPath) {
  int eqTerm = 0, coefs[2];

  lateralCoeficients(sideComb, coefs);

  eqTerm = coefs[0] * mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() +
           coefs[1] * mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset();

  if (debug_)
    std::cout << "DTp2:eqMainBXTerm \t\t\t\t\t In eqMainBXTerm EQTerm(BX): " << eqTerm << std::endl;

  return (eqTerm);
}

/**
 * Evalúa la suma característica de cada par de celdas, según la lateralidad
 * de la trayectoria. Semejante a la anterior, pero aplica las correcciones
 * debidas a los retardos de la electrónica, junto con la del Bunch Crossing
 *
 * El orden de los índices de capa es crítico:
 *    layerIdx[0] -> Capa más baja,
 *    layerIdx[1] -> Capa más alta
 */
int MuonPathAnalyzerPerSL::eqMainTerm(LATERAL_CASES sideComb[2], int layerIdx[2], MuonPath *mPath, int bxValue) {
  int eqTerm = 0, coefs[2];

  lateralCoeficients(sideComb, coefs);

  if (!use_LSB_)
    eqTerm = coefs[0] * (mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() - bxValue) +
             coefs[1] * (mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset() - bxValue);
  else
    eqTerm = coefs[0] * floor((DRIFT_SPEED / (10 * x_precision_)) *
                              (mPath->primitive(layerIdx[0])->tdcTimeStampNoOffset() - bxValue)) +
             coefs[1] * floor((DRIFT_SPEED / (10 * x_precision_)) *
                              (mPath->primitive(layerIdx[1])->tdcTimeStampNoOffset() - bxValue));

  if (debug_)
    std::cout << "DTp2:\t\t\t\t\t EQTerm(Main): " << eqTerm << std::endl;

  return (eqTerm);
}

/**
 * Devuelve los coeficientes (+1 ó -1) de lateralidad para un par dado.
 * De momento es útil para poder codificar la nueva funcionalidad en la que se
 * calcula el BX.
 */

void MuonPathAnalyzerPerSL::lateralCoeficients(LATERAL_CASES sideComb[2], int *coefs) {
  if ((sideComb[0] == LEFT) && (sideComb[1] == LEFT)) {
    *(coefs) = +1;
    *(coefs + 1) = -1;
  } else if ((sideComb[0] == LEFT) && (sideComb[1] == RIGHT)) {
    *(coefs) = +1;
    *(coefs + 1) = +1;
  } else if ((sideComb[0] == RIGHT) && (sideComb[1] == LEFT)) {
    *(coefs) = -1;
    *(coefs + 1) = -1;
  } else if ((sideComb[0] == RIGHT) && (sideComb[1] == RIGHT)) {
    *(coefs) = -1;
    *(coefs + 1) = +1;
  }
}

/**
 * Determines if all valid partial lateral combinations share the same value
 * of 'bxValue'.
 */
bool MuonPathAnalyzerPerSL::sameBXValue(PARTIAL_LATQ_TYPE *latq) {
  bool result = true;
  /*
      Para evitar los errores de precision en el cálculo, en vez de forzar un
      "igual" estricto a la hora de comparar los diferentes valores de BX, se
      obliga a que la diferencia entre pares sea menor que un cierto valor umbral.
      Para hacerlo cómodo se crean 6 booleanos que evalúan cada posible diferencia
    */

  if (debug_)
    std::cout << "Dtp2:sameBXValue bxTolerance_: " << bxTolerance_ << std::endl;

  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d01:" << abs(latq[0].bxValue - latq[1].bxValue) << std::endl;
  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d02:" << abs(latq[0].bxValue - latq[2].bxValue) << std::endl;
  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d03:" << abs(latq[0].bxValue - latq[3].bxValue) << std::endl;
  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d12:" << abs(latq[1].bxValue - latq[2].bxValue) << std::endl;
  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d13:" << abs(latq[1].bxValue - latq[3].bxValue) << std::endl;
  if (debug_)
    std::cout << "Dtp2:sameBXValue \t\t\t\t\t\t d23:" << abs(latq[2].bxValue - latq[3].bxValue) << std::endl;

  bool d01, d02, d03, d12, d13, d23;
  d01 = (abs(latq[0].bxValue - latq[1].bxValue) <= bxTolerance_) ? true : false;
  d02 = (abs(latq[0].bxValue - latq[2].bxValue) <= bxTolerance_) ? true : false;
  d03 = (abs(latq[0].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;
  d12 = (abs(latq[1].bxValue - latq[2].bxValue) <= bxTolerance_) ? true : false;
  d13 = (abs(latq[1].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;
  d23 = (abs(latq[2].bxValue - latq[3].bxValue) <= bxTolerance_) ? true : false;

  /* Casos con 4 grupos de combinaciones parciales de lateralidad validas */
  if ((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d01 && d12 && d23))
    result = false;
  else
      /* Los 4 casos posibles de 3 grupos de lateralidades parciales validas */
      if (((latq[0].latQValid && latq[1].latQValid && latq[2].latQValid) && !(d01 && d12)) or
          ((latq[0].latQValid && latq[1].latQValid && latq[3].latQValid) && !(d01 && d13)) or
          ((latq[0].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d02 && d23)) or
          ((latq[1].latQValid && latq[2].latQValid && latq[3].latQValid) && !(d12 && d23)))
    result = false;
  else
      /* Por ultimo, los 6 casos posibles de pares de lateralidades parciales validas */

      if (((latq[0].latQValid && latq[1].latQValid) && !d01) or ((latq[0].latQValid && latq[2].latQValid) && !d02) or
          ((latq[0].latQValid && latq[3].latQValid) && !d03) or ((latq[1].latQValid && latq[2].latQValid) && !d12) or
          ((latq[1].latQValid && latq[3].latQValid) && !d13) or ((latq[2].latQValid && latq[3].latQValid) && !d23))
    result = false;

  return result;
}

/** Calcula los parámetros de la(s) trayectoria(s) detectadas.
 *
 * Asume que el origen de coordenadas está en al lado 'izquierdo' de la cámara
 * con el eje 'X' en la posición media vertical de todas las celdas.
 * El eje 'Y' se apoya sobre los hilos de las capas 1 y 3 y sobre los costados
 * de las capas 0 y 2.
 */
void MuonPathAnalyzerPerSL::calculatePathParameters(MuonPath *mPath) {
  // El orden es importante. No cambiar sin revisar el codigo.
  if (debug_)
    std::cout << "DTp2:calculatePathParameters \t\t\t\t\t\t  calculating calcCellDriftAndXcoor(mPath) " << std::endl;
  calcCellDriftAndXcoor(mPath);
  //calcTanPhiXPosChamber(mPath);
  if (debug_)
    std::cout << "DTp2:calculatePathParameters \t\t\t\t\t\t  checking mPath->quality() " << mPath->quality()
              << std::endl;
  if (mPath->quality() == HIGHQ or mPath->quality() == HIGHQGHOST) {
    if (debug_)
      std::cout
          << "DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test passed, now calcTanPhiXPosChamber4Hits(mPath) "
          << std::endl;
    calcTanPhiXPosChamber4Hits(mPath);
  } else {
    if (debug_)
      std::cout
          << "DTp2:calculatePathParameters \t\t\t\t\t\t\t  Quality test NOT passed calcTanPhiXPosChamber3Hits(mPath) "
          << std::endl;
    calcTanPhiXPosChamber3Hits(mPath);
  }

  if (debug_)
    std::cout << "DTp2:calculatePathParameters \t\t\t\t\t\t calcChiSquare(mPath) " << std::endl;
  calcChiSquare(mPath);
}

void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber(MuonPath *mPath) {
  /*
      La mayoría del código de este método tiene que ser optimizado puesto que
      se hacen llamadas y cálculos redundantes que ya se han evaluado en otros
      métodos previos.

      Hay que hacer una revisión de las ecuaciones para almacenar en el 'MuonPath'
      una serie de parámetro característicos (basados en sumas y productos, para
      que su implementación en FPGA sea sencilla) con los que, al final del
      proceso, se puedan calcular el ángulo y la coordenada horizontal.

      De momento se deja este código funcional extraído directamente de las
      ecuaciones de la recta.
    */
  int layerIdx[2];
  /*
      To calculate path's angle are only necessary two valid primitives.
      This method should be called only when a 'MuonPath' is determined as valid,
      so, at least, three of its primitives must have a valid time.
      With this two comparitions (which can be implemented easily as multiplexors
      in the FPGA) this method ensures to catch two of those valid primitives to
      evaluate the angle.

      The first one is below the middle line of the superlayer, while the other
      one is above this line
    */
  if (mPath->primitive(0)->isValidTime())
    layerIdx[0] = 0;
  else
    layerIdx[0] = 1;

  if (mPath->primitive(3)->isValidTime())
    layerIdx[1] = 3;
  else
    layerIdx[1] = 2;

  /* We identify along which cells' sides the muon travels */
  LATERAL_CASES sideComb[2];
  sideComb[0] = (mPath->lateralComb())[layerIdx[0]];
  sideComb[1] = (mPath->lateralComb())[layerIdx[1]];

  /* Horizontal gap between cells in cell's semi-length units */
  int dHoriz = (mPath->cellLayout())[layerIdx[1]] - (mPath->cellLayout())[layerIdx[0]];

  /* Vertical gap between cells in cell's height units */
  int dVert = layerIdx[1] - layerIdx[0];

  /*-----------------------------------------------------------------*/
  /*--------------------- Phi angle calculation ---------------------*/
  /*-----------------------------------------------------------------*/
  float num = CELL_SEMILENGTH * dHoriz + DRIFT_SPEED * eqMainTerm(sideComb, layerIdx, mPath, mPath->bxTimeValue());

  float denom = CELL_HEIGHT * dVert;
  float tanPhi = num / denom;

  mPath->setTanPhi(tanPhi);

  /*-----------------------------------------------------------------*/
  /*----------------- Horizontal coord. calculation -----------------*/
  /*-----------------------------------------------------------------*/

  /*
      Using known coordinates, relative to superlayer axis reference, (left most
      superlayer side, and middle line between 2nd and 3rd layers), calculating
      horizontal coordinate implies using a basic line equation:
      (y - y0) = (x - x0) * cotg(Phi)
      This horizontal coordinate can be obtained setting y = 0 on last equation,
      and also setting y0 and x0 with the values of a known muon's path cell
      position hit.
      It's enough to use the lower cell (layerIdx[0]) coordinates. So:
      xC = x0 - y0 * tan(Phi)
    */
  float lowerXPHorizPos = mPath->xCoorCell(layerIdx[0]);

  float lowerXPVertPos = 0;  // This is only the absolute value distance.
  if (layerIdx[0] == 0)
    lowerXPVertPos = CELL_HEIGHT + CELL_SEMIHEIGHT;
  else
    lowerXPVertPos = CELL_SEMIHEIGHT;

  mPath->setHorizPos(lowerXPHorizPos + lowerXPVertPos * tanPhi);
}

/**
 * Cálculos de coordenada y ángulo para un caso de 4 HITS de alta calidad.
 */
void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber4Hits(MuonPath *mPath) {
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));
  int numerator = 3 * (int)round(mPath->xCoorCell(3) / (10 * x_precision_)) +
                  (int)round(mPath->xCoorCell(2) / (10 * x_precision_)) -
                  (int)round(mPath->xCoorCell(1) / (10 * x_precision_)) -
                  3 * (int)round(mPath->xCoorCell(0) / (10 * x_precision_));
  int CELL_HEIGHT_JM = pow(2, 15) / ((int)(10 * CELL_HEIGHT));
  int tanPhi_x4096 = (numerator * CELL_HEIGHT_JM) >> (3 + numberOfBits);
  mPath->setTanPhi(tanPhi_x4096 * tanPsi_precision_);

  float XPos = (mPath->xCoorCell(0) + mPath->xCoorCell(1) + mPath->xCoorCell(2) + mPath->xCoorCell(3)) / 4;
  mPath->setHorizPos(floor(XPos / (10 * x_precision_)) * 10 * x_precision_);
}

/**
 * Cálculos de coordenada y ángulo para un caso de 3 HITS.
 */
void MuonPathAnalyzerPerSL::calcTanPhiXPosChamber3Hits(MuonPath *mPath) {
  int layerIdx[2];
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));

  if (mPath->primitive(0)->isValidTime())
    layerIdx[0] = 0;
  else
    layerIdx[0] = 1;

  if (mPath->primitive(3)->isValidTime())
    layerIdx[1] = 3;
  else
    layerIdx[1] = 2;

  /*-----------------------------------------------------------------*/
  /*--------------------- Phi angle calculation ---------------------*/
  /*-----------------------------------------------------------------*/

  int tan_division_denominator_bits = 16;

  int num =
      ((int)((int)(x_prec_inv * mPath->xCoorCell(layerIdx[1])) - (int)(x_prec_inv * mPath->xCoorCell(layerIdx[0])))
       << (12 - numberOfBits));
  int denominator = (layerIdx[1] - layerIdx[0]) * CELL_HEIGHT;
  int denominator_inv = ((int)(0.5 + pow(2, tan_division_denominator_bits) / float(denominator)));

  float tanPhi = ((num * denominator_inv) >> tan_division_denominator_bits) / ((1. / tanPsi_precision_));

  mPath->setTanPhi(tanPhi);

  /*-----------------------------------------------------------------*/
  /*----------------- Horizontal coord. calculation -----------------*/
  /*-----------------------------------------------------------------*/
  float XPos = 0;
  if (mPath->primitive(0)->isValidTime() and mPath->primitive(3)->isValidTime())
    XPos = (mPath->xCoorCell(0) + mPath->xCoorCell(3)) / 2;
  else
    XPos = (mPath->xCoorCell(1) + mPath->xCoorCell(2)) / 2;

  mPath->setHorizPos(floor(XPos / (10 * x_precision_)) * 10 * x_precision_);
}

/**
 * Calcula las distancias de deriva respecto de cada "wire" y la posición
 * horizontal del punto de interacción en cada celda respecto del sistema
 * de referencia de la cámara.
 *
 * La posición horizontal de cada hilo es calculada en el "DTPrimitive".
 */
void MuonPathAnalyzerPerSL::calcCellDriftAndXcoor(MuonPath *mPath) {
  long int drift_speed_new = 889;
  //long int drift_speed_new = (long int) (round (DRIFT_SPEED * 1000 * 10.24) / (10*x_precision_*10) );
  long int drift_dist_um_x4;
  long int wireHorizPos_x4;
  long int pos_mm_x4;
  int x_prec_inv = (int)(1. / (10. * x_precision_));

  for (int i = 0; i <= 3; i++)
    if (mPath->primitive(i)->isValidTime()) {
      drift_dist_um_x4 =
          drift_speed_new * ((long int)mPath->primitive(i)->tdcTimeStampNoOffset() - (long int)mPath->bxTimeValue());
      wireHorizPos_x4 = (long)(mPath->primitive(i)->wireHorizPos() * x_prec_inv);

      if ((mPath->lateralComb())[i] == LEFT)
        pos_mm_x4 = wireHorizPos_x4 - (drift_dist_um_x4 >> 10);
      else
        pos_mm_x4 = wireHorizPos_x4 + (drift_dist_um_x4 >> 10);

      mPath->setXCoorCell(pos_mm_x4 * (10 * x_precision_), i);
      mPath->setDriftDistance(((float)(drift_dist_um_x4 >> 10)) * (10 * x_precision_), i);
    }
}

/**
 * Calcula el estimador de calidad de la trayectoria.
 */
void MuonPathAnalyzerPerSL::calcChiSquare(MuonPath *mPath) {
  /*
    float xi, zi, factor;

    float chi = 0;
    float mu  = mPath->tanPhi();
    float b   = mPath->horizPos();

    const float baseWireYPos = -1.5 * CELL_HEIGHT;

    for (int i = 0; i <= 3; i++)
	if ( mPath->primitive(i)->isValidTime() ) {
	    zi = baseWireYPos + CELL_HEIGHT * i;
	    xi = mPath->xCoorCell(i);

	    factor = xi - mu*zi - b;
	    chi += (factor * factor);
	}
*/
  int x_prec_inv = (int)(1. / (10. * x_precision_));
  int numberOfBits = (int)(round(std::log(x_prec_inv) / std::log(2.)));
  long int Z_FACTOR[4] = {-6, -2, 2, 6};
  for (int i = 0; i < 4; i++) {
    Z_FACTOR[i] = Z_FACTOR[i] * (long int)CELL_HEIGHT;
  }
  long int sum_A = 0, sum_B = 0;
  long int chi2_mm2_x1024 = 0;
  for (int i = 0; i < 4; i++) {
    if (mPath->primitive(i)->isValidTime()) {
      sum_A = (((int)(mPath->xCoorCell(i) / (10 * x_precision_))) - ((int)(mPath->horizPos() / (10 * x_precision_))))
              << (14 - numberOfBits);
      sum_B = Z_FACTOR[i] * ((int)(mPath->tanPhi() / tanPsi_precision_));
      chi2_mm2_x1024 += (sum_A - sum_B) * (sum_A - sum_B);
      //cout << "sum_A=" << sum_A << " sum_B=" << sum_B << " pow((sum_A - sum_B),2)=" << pow((sum_A - sum_B),2) << " chi2_mm2_x1024=" << chi2_mm2_x1024 << endl;
    }
  }
  chi2_mm2_x1024 = chi2_mm2_x1024 >> 18;
  //cout << "chi2_mm2_x1024=" << chi2_mm2_x1024 << " chi2_mm2_x1024 / 1024.=" << chi2_mm2_x1024 / 1024. << endl;

  mPath->setChiSquare(((double)chi2_mm2_x1024 / 1024.));
  //mPath->setChiSquare(0.);
}

/**
 * Este método devuelve cual layer no se está utilizando en el
 * 'LAYER_ARRANGEMENT' cuyo índice se pasa como parámetro.
 * 
 * ¡¡¡ OJO !!! Este método es completamente dependiente de esa macro.
 * Si hay cambios en ella, HAY QUE CAMBIAR EL MÉTODO.
 * 
 *  LAYER_ARRANGEMENTS[MAX_VERT_ARRANG][3] = {
 *    {0, 1, 2}, {1, 2, 3},                       // Grupos consecutivos
 *    {0, 1, 3}, {0, 2, 3}                        // Grupos salteados
 *  };
 */
int MuonPathAnalyzerPerSL::omittedHit(int idx) {
  int ans = -1;

  switch (idx) {
    case 0:
      ans = 3;
      break;
    case 1:
      ans = 0;
      break;
    case 2:
      ans = 2;
      break;
    case 3:
      ans = 1;
      break;
  }

  return ans;
}
