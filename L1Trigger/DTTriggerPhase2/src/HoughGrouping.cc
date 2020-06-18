#include "L1Trigger/DTTriggerPhase2/interface/HoughGrouping.h"

using namespace std;
using namespace edm;
using namespace cmsdt;

namespace {
  struct {
    bool operator()(ProtoCand a, ProtoCand b) const {
      unsigned short int sumhqa = 0;
      unsigned short int sumhqb = 0;
      unsigned short int sumlqa = 0;
      unsigned short int sumlqb = 0;
      double sumdista = 0;
      double sumdistb = 0;

      for (unsigned short int lay = 0; lay < 8; lay++) {
        sumhqa += (unsigned short int)a.isThereHitInLayer_[lay];
        sumhqb += (unsigned short int)b.isThereHitInLayer_[lay];
        sumlqa += (unsigned short int)a.isThereNeighBourHitInLayer_[lay];
        sumlqb += (unsigned short int)b.isThereNeighBourHitInLayer_[lay];
        sumdista += a.xDistToPattern_[lay];
        sumdistb += b.xDistToPattern_[lay];
      }

      if (a.nLayersWithHits_ != b.nLayersWithHits_)
        return (a.nLayersWithHits_ > b.nLayersWithHits_);  // number of layers with hits
      else if (sumhqa != sumhqb)
        return (sumhqa > sumhqb);  // number of hq hits
      else if (sumlqa != sumlqb)
        return (sumlqa > sumlqb);  // number of lq hits
      else if (a.nHitsDiff_ != b.nHitsDiff_)
        return (a.nHitsDiff_ < b.nHitsDiff_);  // abs. diff. between SL1 & SL3 hits
      else
        return (sumdista < sumdistb);  // abs. dist. to digis
    }
  } HoughOrdering;
}  // namespace
// ============================================================================
// Constructors and destructor
// ============================================================================
HoughGrouping::HoughGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC) : MotherGrouping(pset, iC) {
  // Obtention of parameters
  debug_ = pset.getUntrackedParameter<bool>("debug");
  if (debug_)
    cout << "HoughGrouping: constructor" << endl;

  // HOUGH TRANSFORM CONFIGURATION
  angletan_ = pset.getUntrackedParameter<double>("angletan");
  anglebinwidth_ = pset.getUntrackedParameter<double>("anglebinwidth");
  posbinwidth_ = pset.getUntrackedParameter<double>("posbinwidth");

  // MAXIMA SEARCH CONFIGURATION
  maxdeltaAngDeg_ = pset.getUntrackedParameter<double>("maxdeltaAngDeg");
  maxdeltaPos_ = pset.getUntrackedParameter<double>("maxdeltaPos");
  upperNumber_ = (unsigned short int)pset.getUntrackedParameter<int>("UpperNumber");
  lowerNumber_ = (unsigned short int)pset.getUntrackedParameter<int>("LowerNumber");

  // HITS ASSOCIATION CONFIGURATION
  maxDistanceToWire_ = pset.getUntrackedParameter<double>("MaxDistanceToWire");

  // CANDIDATE QUALITY CONFIGURATION
  minNLayerHits_ = (unsigned short int)pset.getUntrackedParameter<int>("minNLayerHits");
  minSingleSLHitsMax_ = (unsigned short int)pset.getUntrackedParameter<int>("minSingleSLHitsMax");
  minSingleSLHitsMin_ = (unsigned short int)pset.getUntrackedParameter<int>("minSingleSLHitsMin");
  allowUncorrelatedPatterns_ = pset.getUntrackedParameter<bool>("allowUncorrelatedPatterns");
  minUncorrelatedHits_ = (unsigned short int)pset.getUntrackedParameter<int>("minUncorrelatedHits");

  dtGeomH = iC.esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();
}

HoughGrouping::~HoughGrouping() {
  if (debug_)
    cout << "HoughGrouping: destructor" << endl;
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void HoughGrouping::initialise(const edm::EventSetup& iEventSetup) {
  if (debug_)
    cout << "HoughGrouping::initialise" << endl;

  resetAttributes();

  maxrads_ = TMath::PiOver2() - TMath::ATan(angletan_);
  minangle_ = anglebinwidth_ * TMath::TwoPi() / 360;
  halfanglebins_ = TMath::Nint(maxrads_ / minangle_ + 1);
  anglebins_ = (unsigned short int)2 * halfanglebins_;
  oneanglebin_ = maxrads_ / halfanglebins_;

  maxdeltaAng_ = maxdeltaAngDeg_ * TMath::TwoPi() / 360;

  // Initialisation of anglemap. Posmap depends on the size of the chamber.
  double phi = 0;
  anglemap_ = {};
  for (unsigned short int ab = 0; ab < halfanglebins_; ab++) {
    anglemap_[ab] = phi;
    phi += oneanglebin_;
  }

  phi = (TMath::Pi() - maxrads_);
  for (unsigned short int ab = halfanglebins_; ab < anglebins_; ab++) {
    anglemap_[ab] = phi;
    phi += oneanglebin_;
  }

  linespace_ = new unsigned short int*[anglebins_];

  if (debug_) {
    cout << "\nHoughGrouping::ResetAttributes - Information from the initialisation of HoughGrouping:" << endl;
    cout << "HoughGrouping::ResetAttributes - maxrads: " << maxrads_ << endl;
    cout << "HoughGrouping::ResetAttributes - anglebinwidth: " << anglebinwidth_ << endl;
    cout << "HoughGrouping::ResetAttributes - minangle: " << minangle_ << endl;
    cout << "HoughGrouping::ResetAttributes - halfanglebins: " << halfanglebins_ << endl;
    cout << "HoughGrouping::ResetAttributes - anglebins: " << anglebins_ << endl;
    cout << "HoughGrouping::ResetAttributes - oneanglebin: " << oneanglebin_ << endl;
    cout << "HoughGrouping::ResetAttributes - posbinwidth: " << posbinwidth_ << endl;
  }

  const MuonGeometryRecord& geom = iEventSetup.get<MuonGeometryRecord>();
  dtGeo_ = &geom.get(dtGeomH);
}

void HoughGrouping::run(edm::Event& iEvent,
                        const edm::EventSetup& iEventSetup,
                        const DTDigiCollection& digis,
                        MuonPathPtrs& outMpath) {
  if (debug_)
    cout << "\nHoughGrouping::run" << endl;

  resetAttributes();

  if (debug_)
    cout << "HoughGrouping::run - Beginning digis' loop..." << endl;
  LocalPoint wirePosInLay, wirePosInChamber;
  GlobalPoint wirePosGlob;
  for (DTDigiCollection::DigiRangeIterator dtLayerIdIt = digis.begin(); dtLayerIdIt != digis.end(); dtLayerIdIt++) {
    const DTLayer* lay = dtGeo_->layer((*dtLayerIdIt).first);
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerIdIt).second).first;
         digiIt != ((*dtLayerIdIt).second).second;
         digiIt++) {
      if (debug_) {
        cout << "\nHoughGrouping::run - Digi number " << idigi_ << endl;
        cout << "HoughGrouping::run - Wheel: " << (*dtLayerIdIt).first.wheel() << endl;
        cout << "HoughGrouping::run - Chamber: " << (*dtLayerIdIt).first.station() << endl;
        cout << "HoughGrouping::run - Sector: " << (*dtLayerIdIt).first.sector() << endl;
        cout << "HoughGrouping::run - Superlayer: " << (*dtLayerIdIt).first.superLayer() << endl;
        cout << "HoughGrouping::run - Layer: " << (*dtLayerIdIt).first.layer() << endl;
        cout << "HoughGrouping::run - Wire: " << (*digiIt).wire() << endl;
        cout << "HoughGrouping::run - First wire: " << lay->specificTopology().firstChannel() << endl;
        cout << "HoughGrouping::run - Last wire: " << lay->specificTopology().lastChannel() << endl;
        cout << "HoughGrouping::run - First wire x: "
             << lay->specificTopology().wirePosition(lay->specificTopology().firstChannel()) << endl;
        cout << "HoughGrouping::run - Last wire x: "
             << lay->specificTopology().wirePosition(lay->specificTopology().lastChannel()) << endl;
        cout << "HoughGrouping::run - Cell width: " << lay->specificTopology().cellWidth() << endl;
        cout << "HoughGrouping::run - Cell height: " << lay->specificTopology().cellHeight() << endl;
      }
      if ((*dtLayerIdIt).first.superLayer() == 2)
        continue;

      wirePosInLay = LocalPoint(lay->specificTopology().wirePosition((*digiIt).wire()), 0, 0);
      wirePosGlob = lay->toGlobal(wirePosInLay);
      wirePosInChamber = lay->chamber()->toLocal(wirePosGlob);

      if ((*dtLayerIdIt).first.superLayer() == 3) {
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()] = DTPrimitive();
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setTDCTimeStamp((*digiIt).time());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap_[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      } else {
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()] = DTPrimitive();
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setTDCTimeStamp((*digiIt).time());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap_[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      }

      // Obtaining geometrical info of the chosen chamber
      if (xlowlim_ == 0 && xhighlim_ == 0 && zlowlim_ == 0 && zhighlim_ == 0) {
        thewheel_ = (*dtLayerIdIt).first.wheel();
        thestation_ = (*dtLayerIdIt).first.station();
        thesector_ = (*dtLayerIdIt).first.sector();
        obtainGeometricalBorders(lay);
      }

      if (debug_) {
        cout << "HoughGrouping::run - X position of the cell (chamber frame of reference): " << wirePosInChamber.x()
             << endl;
        cout << "HoughGrouping::run - Y position of the cell (chamber frame of reference): " << wirePosInChamber.y()
             << endl;
        cout << "HoughGrouping::run - Z position of the cell (chamber frame of reference): " << wirePosInChamber.z()
             << endl;
      }

      hitvec_.push_back({wirePosInChamber.x() - 1.05, wirePosInChamber.z()});
      hitvec_.push_back({wirePosInChamber.x() + 1.05, wirePosInChamber.z()});
      nhits_ += 2;

      idigi_++;
    }
  }

  if (debug_) {
    cout << "\nHoughGrouping::run - nhits: " << nhits_ << endl;
    cout << "HoughGrouping::run - idigi: " << idigi_ << endl;
  }

  if (hitvec_.size() == 0) {
    if (debug_)
      cout << "HoughGrouping::run - No digis present in this chamber." << endl;
    return;
  }

  // Perform the Hough transform of the inputs.
  doHoughTransform();

  // Obtain the maxima
  maxima_ = getMaximaVector();
  resetPosElementsOfLinespace();

  if (maxima_.size() == 0) {
    if (debug_)
      cout << "HoughGrouping::run - No good maxima found in this event." << endl;
    return;
  }

  DTChamberId TheChambId(thewheel_, thestation_, thesector_);
  const DTChamber* TheChamb = dtGeo_->chamber(TheChambId);
  std::vector<ProtoCand> cands;

  for (unsigned short int ican = 0; ican < maxima_.size(); ican++) {
    if (debug_)
      cout << "\nHoughGrouping::run - Candidate number: " << ican << endl;
    cands.push_back(associateHits(TheChamb, maxima_.at(ican).first, maxima_.at(ican).second));
  }

  // Now we filter them:
  orderAndFilter(cands, outMpath);
  if (debug_)
    cout << "HoughGrouping::run - now we have our muonpaths! It has " << outMpath.size() << " elements" << endl;

  cands.clear();
  //  short int indi = cands.size() - 1;
  //  while (!cands.empty()) {
  //    delete[] get<1>(cands.at(indi));
  //    delete[] get<2>(cands.at(indi));
  //    delete[] get<4>(cands.at(indi));
  //    delete[] get<5>(cands.at(indi));
  //
  //    cands.pop_back();
  //    indi--;
  //  }
  //  std::vector<std::tuple<unsigned short int, bool*, bool*, unsigned short int, double*, DTPrimitive*>>().swap(cands);
  return;
}

void HoughGrouping::finish() {
  if (debug_)
    cout << "HoughGrouping::finish" << endl;
  return;
}

// ============================================================================
// Other methods
// ============================================================================
void HoughGrouping::resetAttributes() {
  if (debug_)
    cout << "HoughGrouping::ResetAttributes" << endl;
  // std::vector's:
  maxima_.clear();
  //   cands.clear();
  hitvec_.clear();

  // Integer-type variables:
  spacebins_ = 0;
  idigi_ = 0;
  nhits_ = 0;
  xlowlim_ = 0;
  xhighlim_ = 0;
  zlowlim_ = 0;
  zhighlim_ = 0;
  thestation_ = 0;
  thesector_ = 0;
  thewheel_ = 0;

  // Arrays:
  // NOTE: linespace array is treated and reset separately

  // Maps (dictionaries):
  posmap_.clear();
  for (unsigned short int abslay = 0; abslay < 8; abslay++)
    digimap_[abslay].clear();
}

void HoughGrouping::resetPosElementsOfLinespace() {
  if (debug_)
    cout << "HoughGrouping::ResetPosElementsOfLinespace" << endl;
  for (unsigned short int ab = 0; ab < anglebins_; ab++) {
    delete[] linespace_[ab];
  }
}

void HoughGrouping::obtainGeometricalBorders(const DTLayer* lay) {
  if (debug_)
    cout << "HoughGrouping::ObtainGeometricalBorders" << endl;
  LocalPoint FirstWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(
                                lay->chamber()->superLayer(1)->layer(1)->specificTopology().firstChannel()),
                            0,
                            0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobal = lay->chamber()->superLayer(1)->layer(1)->toGlobal(FirstWireLocal);
  LocalPoint FirstWireLocalCh = lay->chamber()->toLocal(FirstWireGlobal);

  LocalPoint LastWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(
                               lay->chamber()->superLayer(1)->layer(1)->specificTopology().lastChannel()),
                           0,
                           0);
  GlobalPoint LastWireGlobal = lay->chamber()->superLayer(1)->layer(1)->toGlobal(LastWireLocal);
  LocalPoint LastWireLocalCh = lay->chamber()->toLocal(LastWireGlobal);

  //   unsigned short int upsl = thestation == 4 ? 2 : 3;
  unsigned short int upsl = thestation_ == 4 ? 3 : 3;
  if (debug_)
    cout << "HoughGrouping::ObtainGeometricalBorders - uppersuperlayer: " << upsl << endl;

  LocalPoint FirstWireLocalUp(lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().wirePosition(
                                  lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().firstChannel()),
                              0,
                              0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobalUp = lay->chamber()->superLayer(upsl)->layer(4)->toGlobal(FirstWireLocalUp);
  LocalPoint FirstWireLocalChUp = lay->chamber()->toLocal(FirstWireGlobalUp);

  xlowlim_ = FirstWireLocalCh.x() - lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  xhighlim_ = LastWireLocalCh.x() + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  zlowlim_ = FirstWireLocalChUp.z() - lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().cellHeight() / 2;
  zhighlim_ = LastWireLocalCh.z() + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellHeight() / 2;

  spacebins_ = TMath::Nint(TMath::Abs(xhighlim_ - xlowlim_) / posbinwidth_);
}

void HoughGrouping::doHoughTransform() {
  if (debug_)
    cout << "HoughGrouping::DoHoughTransform" << endl;
  // First we want to obtain the number of bins in angle that we want. To do so, we will consider at first a maximum angle of
  // (in rad.) pi/2 - arctan(0.3) (i.e. ~73º) and a resolution (width of bin angle) of 2º.

  if (debug_) {
    cout << "HoughGrouping::DoHoughTransform - maxrads: " << maxrads_ << endl;
    cout << "HoughGrouping::DoHoughTransform - minangle: " << minangle_ << endl;
    cout << "HoughGrouping::DoHoughTransform - halfanglebins: " << halfanglebins_ << endl;
    cout << "HoughGrouping::DoHoughTransform - anglebins: " << anglebins_ << endl;
    cout << "HoughGrouping::DoHoughTransform - oneanglebin: " << oneanglebin_ << endl;
    cout << "HoughGrouping::DoHoughTransform - spacebins: " << spacebins_ << endl;
  }

  double rho = 0, phi = 0, sbx = 0;
  // lowinitsb defines the center of the first bin in the distance dimension
  //   double lowinitsb = xlowlim_;
  double lowinitsb = xlowlim_ + posbinwidth_ / 2;

  // Initialisation
  for (unsigned short int ab = 0; ab < anglebins_; ab++) {
    linespace_[ab] = new unsigned short int[spacebins_];
    sbx = lowinitsb;
    for (unsigned short int sb = 0; sb < spacebins_; sb++) {
      posmap_[sb] = sbx;
      linespace_[ab][sb] = 0;
      sbx += posbinwidth_;
    }
  }

  // Filling of the double array and actually doing the transform
  for (unsigned short int i = 0; i < hitvec_.size(); i++) {
    for (unsigned short int ab = 0; ab < anglebins_; ab++) {
      phi = anglemap_[ab];
      rho = hitvec_.at(i).first * TMath::Cos(phi) + hitvec_.at(i).second * TMath::Sin(phi);
      sbx = lowinitsb - posbinwidth_ / 2;
      for (unsigned short int sb = 0; sb < spacebins_; sb++) {
        if (rho < sbx) {
          linespace_[ab][sb]++;
          break;
        }
        sbx += posbinwidth_;
      }
    }
  }
}

std::vector<std::pair<double, double>> HoughGrouping::getMaximaVector() {
  if (debug_)
    cout << "HoughGrouping::getMaximaVector" << endl;
  std::vector<std::tuple<double, double, unsigned short int>> tmpvec;
  tmpvec.clear();

  bool flagsearched = false;
  unsigned short int numbertolookat = upperNumber_;

  while (!flagsearched) {
    for (unsigned short int ab = 0; ab < anglebins_; ab++) {
      for (unsigned short int sb = 0; sb < spacebins_; sb++) {
        if (linespace_[ab][sb] >= numbertolookat)
          tmpvec.push_back({anglemap_[ab], posmap_[sb], linespace_[ab][sb]});
      }
    }
    if (((numbertolookat - 1) < lowerNumber_) || (tmpvec.size() > 0))
      flagsearched = true;
    else
      numbertolookat--;
  }

  if (tmpvec.size() == 0) {
    if (debug_)
      cout << "HoughGrouping::GetMaximaVector - No maxima could be found" << endl;
    std::vector<std::pair<double, double>> finalvec;
    finalvec.clear();
    return finalvec;
  } else {
    std::vector<std::pair<double, double>> finalvec = findTheMaxima(tmpvec);

    // And now obtain the values of m and n of the lines.
    for (unsigned short int i = 0; i < finalvec.size(); i++)
      finalvec.at(i) = transformPair(finalvec.at(i));
    return finalvec;
  }
}

std::pair<double, double> HoughGrouping::transformPair(std::pair<double, double> inputpair) {
  if (debug_)
    cout << "HoughGrouping::transformPair" << endl;
  // input: (ang, pos); output: (m, n)
  if (inputpair.first == 0)
    return {1000, -1000 * inputpair.second};
  else
    return {-1. / TMath::Tan(inputpair.first), inputpair.second / TMath::Sin(inputpair.first)};
}

std::vector<std::pair<double, double>> HoughGrouping::findTheMaxima(
    std::vector<std::tuple<double, double, unsigned short int>> inputvec) {
  if (debug_)
    cout << "HoughGrouping::findTheMaxima" << endl;
  bool fullyreduced = false;
  unsigned short int ind = 0;

  std::vector<unsigned short int> chosenvec;
  chosenvec.clear();
  std::vector<std::pair<double, double>> resultvec;
  resultvec.clear();
  std::pair<double, double> finalpair = {};

  if (debug_)
    cout << "HoughGrouping::findTheMaxima - prewhile" << endl;
  while (!fullyreduced) {
    if (debug_) {
      cout << "\nHoughGrouping::findTheMaxima - New iteration" << endl;
      cout << "HoughGrouping::findTheMaxima - inputvec size: " << inputvec.size() << endl;
      cout << "HoughGrouping::findTheMaxima - ind: " << ind << endl;
      cout << "HoughGrouping::findTheMaxima - maximum deltaang: " << maxdeltaAng_
           << " and maximum deltapos: " << maxdeltaPos_ << endl;
    }
    chosenvec.clear();
    //calculate distances and check out the ones that are near
    if (debug_)
      cout << "HoughGrouping::findTheMaxima - Ours have " << get<2>(inputvec.at(ind))
           << " entries, ang.: " << get<0>(inputvec.at(ind)) << " and pos.: " << get<1>(inputvec.at(ind)) << endl;

    for (unsigned short int j = ind + 1; j < inputvec.size(); j++) {
      if (getTwoDelta(inputvec.at(ind), inputvec.at(j)).first <= maxdeltaAng_ &&
          getTwoDelta(inputvec.at(ind), inputvec.at(j)).second <= maxdeltaPos_) {
        chosenvec.push_back(j);
        if (debug_)
          cout << "HoughGrouping::findTheMaxima -     - Adding num.  " << j
               << " with deltaang: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).first
               << ", and deltapos: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).second << " and with "
               << get<2>(inputvec.at(j)) << " entries, ang.: " << get<0>(inputvec.at(j))
               << " and pos.: " << get<1>(inputvec.at(j)) << endl;
      } else if (debug_)
        cout << "HoughGrouping::findTheMaxima -     - Ignoring num. " << j
             << " with deltaang: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).first
             << ", and deltapos: " << getTwoDelta(inputvec.at(ind), inputvec.at(j)).second << " and with "
             << get<2>(inputvec.at(j)) << " entries." << endl;
    }

    if (debug_)
      cout << "HoughGrouping::findTheMaxima - chosenvecsize: " << chosenvec.size() << endl;

    if (chosenvec.size() == 0) {
      if (ind + 1 >= (unsigned short int)inputvec.size())
        fullyreduced = true;
      if ((get<0>(inputvec.at(ind)) <= maxrads_) || (get<0>(inputvec.at(ind)) >= TMath::Pi() - maxrads_))
        resultvec.push_back({get<0>(inputvec.at(ind)), get<1>(inputvec.at(ind))});
      else if (debug_)
        cout << "HoughGrouping::findTheMaxima -     - Candidate dropped due to an excess in angle" << endl;
      ind++;
      continue;
    }

    // Now average them
    finalpair = getAveragePoint(inputvec, ind, chosenvec);

    // Erase the ones you used
    inputvec.erase(inputvec.begin() + ind);
    for (short int j = chosenvec.size() - 1; j > -1; j--) {
      if (debug_)
        cout << "HoughGrouping::findTheMaxima - erasing index: " << chosenvec.at(j) - 1 << endl;
      inputvec.erase(inputvec.begin() + chosenvec.at(j) - 1);
    }

    if (debug_)
      cout << "HoughGrouping::findTheMaxima - inputvec size: " << inputvec.size() << endl;

    // And add the one you calculated:
    if ((finalpair.first <= maxrads_) || (finalpair.first >= TMath::Pi() - maxrads_))
      resultvec.push_back(finalpair);
    else if (debug_)
      cout << "HoughGrouping::findTheMaxima -     - Candidate dropped due to an excess in angle" << endl;

    if (ind + 1 >= (unsigned short int)inputvec.size())
      fullyreduced = true;
    if (debug_)
      cout << "HoughGrouping::findTheMaxima - iteration ends" << endl;
    ind++;
  }
  if (debug_)
    cout << "HoughGrouping::findTheMaxima - postwhile" << endl;
  return resultvec;
}

std::pair<double, double> HoughGrouping::getTwoDelta(std::tuple<double, double, unsigned short int> pair1,
                                                     std::tuple<double, double, unsigned short int> pair2) {
  if (debug_)
    cout << "HoughGrouping::getTwoDelta" << endl;
  return {TMath::Abs(get<0>(pair1) - get<0>(pair2)), TMath::Abs(get<1>(pair1) - get<1>(pair2))};
}

std::pair<double, double> HoughGrouping::getAveragePoint(
    std::vector<std::tuple<double, double, unsigned short int>> inputvec,
    unsigned short int firstindex,
    std::vector<unsigned short int> indexlist) {
  if (debug_)
    cout << "HoughGrouping::getAveragePoint" << endl;
  std::vector<double> xs;
  xs.clear();
  std::vector<double> ys;
  ys.clear();
  std::vector<double> ws;
  ws.clear();
  xs.push_back(get<0>(inputvec.at(firstindex)));
  ys.push_back(get<1>(inputvec.at(firstindex)));
  //   ws.push_back(get<2>(inputvec.at(firstindex)));
  ws.push_back(TMath::Exp(get<2>(inputvec.at(firstindex))));
  for (unsigned short int i = 0; i < indexlist.size(); i++) {
    xs.push_back(get<0>(inputvec.at(indexlist.at(i))));
    ys.push_back(get<1>(inputvec.at(indexlist.at(i))));
    //     ws.push_back(get<2>(inputvec.at(indexlist.at(i))));
    ws.push_back(TMath::Exp(get<2>(inputvec.at(indexlist.at(i)))));
  }
  return {TMath::Mean(xs.begin(), xs.end(), ws.begin()), TMath::Mean(ys.begin(), ys.end(), ws.begin())};
}

ProtoCand HoughGrouping::associateHits(const DTChamber* thechamb, double m, double n) {
  if (debug_)
    cout << "HoughGrouping::associateHits" << endl;
  LocalPoint tmpLocal, AWireLocal, AWireLocalCh, tmpLocalCh, thepoint;
  GlobalPoint tmpGlobal, AWireGlobal;
  double tmpx = 0;
  double distleft = 0;
  double distright = 0;
  unsigned short int tmpwire = 0;
  unsigned short int abslay = 0;
  LATERAL_CASES lat = NONE;
  bool isleft = false;
  bool isright = false;

  ProtoCand returnPC;
  for (auto l = 0; l < 8; l++) {
    returnPC.isThereHitInLayer_.push_back(false);
    returnPC.isThereNeighBourHitInLayer_.push_back(false);
    returnPC.xDistToPattern_.push_back(0);
    returnPC.dtHits_.push_back(DTPrimitive());
  }

  if (debug_)
    cout << "HoughGrouping::associateHits - Beginning SL loop" << endl;
  for (unsigned short int sl = 1; sl < 3 + 1; sl++) {
    if (sl == 2)
      continue;
    if (debug_)
      cout << "HoughGrouping::associateHits - SL: " << sl << endl;

    for (unsigned short int l = 1; l < 4 + 1; l++) {
      if (debug_)
        cout << "HoughGrouping::associateHits - L: " << l << endl;
      isleft = false;
      isright = false;
      lat = NONE;
      distleft = 0;
      distright = 0;
      if (sl == 1)
        abslay = l - 1;
      else
        abslay = l + 3;
      AWireLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(
                                  thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel()),
                              0,
                              0);
      AWireGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(AWireLocal);
      AWireLocalCh = thechamb->toLocal(AWireGlobal);
      tmpx = (AWireLocalCh.z() - n) / m;

      if ((tmpx <= xlowlim_) || (tmpx >= xhighlim_)) {
        returnPC.dtHits_[abslay] = DTPrimitive();  // empty primitive
        continue;
      }

      thepoint = LocalPoint(tmpx, 0, AWireLocalCh.z());
      tmpwire = thechamb->superLayer(sl)->layer(l)->specificTopology().channel(thepoint);
      if (debug_)
        cout << "HoughGrouping::associateHits - Wire number: " << tmpwire << endl;
      if (debug_)
        cout << "HoughGrouping::associateHits - First channel in layer: "
             << thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel() << endl;
      if ((digimap_[abslay]).count(tmpwire)) {
        // OK, we have a digi, let's choose the laterality, if we can:
        tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire), 0, 0);
        tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
        tmpLocalCh = thechamb->toLocal(tmpGlobal);

        if (TMath::Abs(tmpLocalCh.x() - thepoint.x()) >= maxDistanceToWire_) {
          // The distance where lateralities are not put is 0.03 cm, which is a conservative threshold for the resolution of the cells.
          if ((tmpLocalCh.x() - thepoint.x()) > 0)
            lat = LEFT;
          else
            lat = RIGHT;
        }

        // Filling info
        returnPC.nLayersWithHits_++;
        returnPC.isThereHitInLayer_[abslay] = true;
        returnPC.isThereNeighBourHitInLayer_[abslay] = true;
        if (lat == LEFT)
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
        else if (lat == RIGHT)
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() + 1.05));
        else
          returnPC.xDistToPattern_[abslay] = abs(tmpx - tmpLocalCh.x());
        returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire]);
        returnPC.dtHits_[abslay].setLaterality(lat);
      } else {
        if (debug_)
          cout << "HoughGrouping::associateHits - No hit in the crossing cell" << endl;
        if ((digimap_[abslay]).count(tmpwire - 1))
          isleft = true;
        if ((digimap_[abslay]).count(tmpwire + 1))
          isright = true;
        if (debug_)
          cout << "HoughGrouping::associateHits - There is in the left: " << (int)isleft << endl;
        if (debug_)
          cout << "HoughGrouping::associateHits - There is in the right: " << (int)isright << endl;

        if ((isleft) && (!isright)) {
          tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() + 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire - 1]);
          returnPC.dtHits_[abslay].setLaterality(RIGHT);

        } else if ((!isleft) && (isright)) {
          tmpLocal = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          tmpGlobal = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;
          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
          returnPC.dtHits_[abslay].setLaterality(LEFT);
        } else if ((isleft) && (isright)) {
          LocalPoint tmpLocal_l =
              LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          GlobalPoint tmpGlobal_l = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_l);
          LocalPoint tmpLocalCh_l = thechamb->toLocal(tmpGlobal_l);

          LocalPoint tmpLocal_r =
              LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          GlobalPoint tmpGlobal_r = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_r);
          LocalPoint tmpLocalCh_r = thechamb->toLocal(tmpGlobal_r);

          distleft = TMath::Abs(thepoint.x() - tmpLocalCh_l.x());
          distright = TMath::Abs(thepoint.x() - tmpLocalCh_r.x());

          // Filling info
          returnPC.nLayersWithHits_++;
          returnPC.isThereNeighBourHitInLayer_[abslay] = true;

          returnPC.xDistToPattern_[abslay] = abs(tmpx - (tmpLocalCh.x() - 1.05));
          returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
          returnPC.dtHits_[abslay].setLaterality(LEFT);

          if (distleft < distright) {
            returnPC.xDistToPattern_[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() + 1.05));
            returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire - 1]);
            returnPC.dtHits_[abslay].setLaterality(RIGHT);
          } else {
            returnPC.xDistToPattern_[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() - 1.05));
            returnPC.dtHits_[abslay] = DTPrimitive(digimap_[abslay][tmpwire + 1]);
            returnPC.dtHits_[abslay].setLaterality(LEFT);
          }
        } else {                                     // case where there are no digis
          returnPC.dtHits_[abslay] = DTPrimitive();  // empty primitive
        }
      }
    }
  }

  setDifferenceBetweenSL(returnPC);
  if (debug_) {
    cout << "HoughGrouping::associateHits - Finishing with the candidate. We have found the following of it:" << endl;
    cout << "HoughGrouping::associateHits - # of layers with hits: " << returnPC.nLayersWithHits_ << endl;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      cout << "HoughGrouping::associateHits - For absolute layer: " << lay << endl;
      cout << "HoughGrouping::associateHits - # of HQ hits: " << returnPC.isThereHitInLayer_[lay] << endl;
      cout << "HoughGrouping::associateHits - # of LQ hits: " << returnPC.isThereNeighBourHitInLayer_[lay] << endl;
    }
    cout << "HoughGrouping::associateHits - Abs. diff. between SL1 and SL3 hits: " << returnPC.nHitsDiff_ << endl;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      cout << "HoughGrouping::associateHits - For absolute layer: " << lay << endl;
      cout << "HoughGrouping::associateHits - Abs. distance to digi: " << returnPC.xDistToPattern_[lay] << endl;
    }
  }
  return returnPC;
}

void HoughGrouping::setDifferenceBetweenSL(ProtoCand& tupl) {
  if (debug_)
    cout << "HoughGrouping::setDifferenceBetweenSL" << endl;
  short int absres = 0;
  for (unsigned short int lay = 0; lay < 8; lay++) {
    if (tupl.dtHits_[lay].channelId() > 0) {
      if (lay <= 3)
        absres++;
      else
        absres--;
    }
  }

  if (absres >= 0)
    tupl.nHitsDiff_ = absres;
  else
    tupl.nHitsDiff_ = (unsigned short int)(-absres);
}

void HoughGrouping::orderAndFilter(std::vector<ProtoCand>& invector, MuonPathPtrs& outMuonPath) {
  if (debug_)
    cout << "HoughGrouping::orderAndFilter" << endl;
  // 0: # of layers with hits.
  // 1: # of hits of high quality (the expected line crosses the cell).
  // 2: # of hits of low quality (the expected line is in a neighbouring cell).
  // 3: absolute diff. between the number of hits in SL1 and SL3.
  // 4: absolute distance to all hits of the segment.
  // 5: DTPrimitive of the candidate.

  std::vector<unsigned short int> elstoremove;
  elstoremove.clear();
  // ordering:
  if (debug_)
    cout << "HoughGrouping::orderAndFilter - First ordering" << endl;
  std::sort(invector.begin(), invector.end(), HoughOrdering);

  // Now filtering:
  unsigned short int ind = 0;
  bool filtered = false;
  if (debug_)
    cout << "HoughGrouping::orderAndFilter - Entering while" << endl;
  while (!filtered) {
    if (debug_)
      cout << "\nHoughGrouping::orderAndFilter - New iteration with ind: " << ind << endl;
    elstoremove.clear();
    for (unsigned short int i = ind + 1; i < invector.size(); i++) {
      if (debug_)
        cout << "HoughGrouping::orderAndFilter - Checking index: " << i << endl;
      for (unsigned short int lay = 0; lay < 8; lay++) {
        if (debug_)
          cout << "HoughGrouping::orderAndFilter - Checking layer number: " << lay << endl;
        if (invector.at(i).dtHits_[lay].channelId() == invector.at(ind).dtHits_[lay].channelId() &&
            invector.at(ind).dtHits_[lay].channelId() != -1) {
          invector.at(i).nLayersWithHits_--;
          invector.at(i).isThereHitInLayer_[lay] = false;
          invector.at(i).isThereNeighBourHitInLayer_[lay] = false;
          setDifferenceBetweenSL(invector.at(i));
          // We check that if its a different laterality, the best candidate of the two of them changes its laterality to not-known (that is, both).
          if (invector.at(i).dtHits_[lay].laterality() != invector.at(ind).dtHits_[lay].laterality())
            invector.at(ind).dtHits_[lay].setLaterality(NONE);

          invector.at(i).dtHits_[lay] = DTPrimitive();
        }
      }
      if (debug_)
        cout << "HoughGrouping::orderAndFilter - Finished checking all the layers, now seeing if we should remove the "
                "candidate"
             << endl;

      if (!areThereEnoughHits(invector.at(i))) {
        if (debug_)
          cout << "HoughGrouping::orderAndFilter - This candidate shall be removed!" << endl;
        elstoremove.push_back((unsigned short int)i);
      }
    }

    if (debug_)
      cout << "HoughGrouping::orderAndFilter - We are gonna erase " << elstoremove.size() << " elements" << endl;

    for (short int el = (elstoremove.size() - 1); el > -1; el--) {
      invector.erase(invector.begin() + elstoremove.at(el));
    }

    if (ind + 1 == (unsigned short int)invector.size())
      filtered = true;
    else
      std::sort(invector.begin() + ind + 1, invector.end(), HoughOrdering);
    ind++;
  }

  // Ultimate filter: if the remaining do not fill the requirements (configurable through pset arguments), they are removed also.
  for (short int el = (invector.size() - 1); el > -1; el--) {
    if (!areThereEnoughHits(invector.at(el))) {
      invector.erase(invector.begin() + el);
    }
  }

  if (invector.size() == 0) {
    if (debug_)
      cout << "HoughGrouping::OrderAndFilter - We do not have candidates with the minimum hits required." << endl;
    return;
  } else if (debug_)
    cout << "HoughGrouping::OrderAndFilter - At the end, we have only " << invector.size() << " good paths!" << endl;

  // Packing dt primitives
  for (unsigned short int i = 0; i < invector.size(); i++) {
    DTPrimitivePtrs ptrPrimitive;
    unsigned short int tmplowfill = 0;
    unsigned short int tmpupfill = 0;
    for (unsigned short int lay = 0; lay < 8; lay++) {
      auto dtAux = DTPrimitivePtr(new DTPrimitive(invector.at(i).dtHits_[lay]));
      ptrPrimitive.push_back(std::move(dtAux));
      if (debug_) {
        cout << "\nHoughGrouping::OrderAndFilter - cameraid: " << ptrPrimitive[lay]->cameraId() << endl;
        cout << "HoughGrouping::OrderAndFilter - channelid (GOOD): " << ptrPrimitive[lay]->channelId() << endl;
        cout << "HoughGrouping::OrderAndFilter - channelid (AM):   " << ptrPrimitive[lay]->channelId() - 1 << endl;
      }
      // Fixing channel ID to AM conventions...
      if (ptrPrimitive[lay]->channelId() != -1)
        ptrPrimitive[lay]->setChannelId(ptrPrimitive[lay]->channelId() - 1);

      if (ptrPrimitive[lay]->cameraId() > 0) {
        if (lay < 4)
          tmplowfill++;
        else
          tmpupfill++;
      }
    }

    auto ptrMuonPath = MuonPathPtr(new MuonPath(ptrPrimitive, tmplowfill, tmpupfill));
    outMuonPath.push_back(ptrMuonPath);
    if (debug_) {
      for (unsigned short int lay = 0; lay < 8; lay++) {
        cout << "HoughGrouping::OrderAndFilter - Final cameraID: " << outMuonPath.back()->primitive(lay)->cameraId()
             << endl;
        cout << "HoughGrouping::OrderAndFilter - Final channelID: " << outMuonPath.back()->primitive(lay)->channelId()
             << endl;
        cout << "HoughGrouping::OrderAndFilter - Final time: " << outMuonPath.back()->primitive(lay)->tdcTimeStamp()
             << endl;
      }
    }
  }
  return;
}

bool HoughGrouping::areThereEnoughHits(ProtoCand& tupl) {
  if (debug_)
    cout << "HoughGrouping::areThereEnoughHits" << endl;
  unsigned short int numhitssl1 = 0;
  unsigned short int numhitssl3 = 0;
  for (unsigned short int lay = 0; lay < 8; lay++) {
    if ((tupl.dtHits_[lay].channelId() > 0) && (lay < 4))
      numhitssl1++;
    else if (tupl.dtHits_[lay].channelId() > 0)
      numhitssl3++;
  }

  if (debug_)
    cout << "HoughGrouping::areThereEnoughHits - Hits in SL1: " << numhitssl1 << endl;
  if (debug_)
    cout << "HoughGrouping::areThereEnoughHits - Hits in SL3: " << numhitssl3 << endl;

  if ((numhitssl1 != 0) && (numhitssl3 != 0)) {  // Correlated candidates
    if ((numhitssl1 + numhitssl3) >= minNLayerHits_) {
      if (numhitssl1 > numhitssl3) {
        return ((numhitssl1 >= minSingleSLHitsMax_) && (numhitssl3 >= minSingleSLHitsMin_));
      } else if (numhitssl3 > numhitssl1) {
        return ((numhitssl3 >= minSingleSLHitsMax_) && (numhitssl1 >= minSingleSLHitsMin_));
      } else
        return true;
    }
  } else if (allowUncorrelatedPatterns_) {  // Uncorrelated candidates
    return ((numhitssl1 + numhitssl3) >= minNLayerHits_);
  } else {
    return false;
  }
  return false;
}
