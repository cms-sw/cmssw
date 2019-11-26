// -*- C++ -*-
//
//
// Package:    CMTRawAnalyzer
#include "DPGAnalysis/HcalTools/interface/CMTRawAnalyzer.h"
//
//
//

CMTRawAnalyzer::CMTRawAnalyzer(const edm::ParameterSet& iConfig) {
  verbosity = iConfig.getUntrackedParameter<int>("Verbosity");
  MAPcreation = iConfig.getUntrackedParameter<int>("MapCreation");
  recordNtuples_ = iConfig.getUntrackedParameter<bool>("recordNtuples");
  maxNeventsInNtuple_ = iConfig.getParameter<int>("maxNeventsInNtuple");
  tok_calib_ = consumes<HcalCalibDigiCollection>(iConfig.getParameter<edm::InputTag>("hcalCalibDigiCollectionTag"));  //
  tok_hbhe_ = consumes<HBHEDigiCollection>(iConfig.getParameter<edm::InputTag>("hbheDigiCollectionTag"));
  tok_ho_ = consumes<HODigiCollection>(iConfig.getParameter<edm::InputTag>("hoDigiCollectionTag"));
  tok_hf_ = consumes<HFDigiCollection>(iConfig.getParameter<edm::InputTag>("hfDigiCollectionTag"));  //
  tok_qie11_ = consumes<QIE11DigiCollection>(iConfig.getParameter<edm::InputTag>("hbheQIE11DigiCollectionTag"));
  tok_qie10_ = consumes<QIE10DigiCollection>(iConfig.getParameter<edm::InputTag>("hbheQIE10DigiCollectionTag"));
  recordHistoes_ = iConfig.getUntrackedParameter<bool>("recordHistoes");
  studyRunDependenceHist_ = iConfig.getUntrackedParameter<bool>("studyRunDependenceHist");
  studyCapIDErrorsHist_ = iConfig.getUntrackedParameter<bool>("studyCapIDErrorsHist");
  studyRMSshapeHist_ = iConfig.getUntrackedParameter<bool>("studyRMSshapeHist");
  studyRatioShapeHist_ = iConfig.getUntrackedParameter<bool>("studyRatioShapeHist");
  studyTSmaxShapeHist_ = iConfig.getUntrackedParameter<bool>("studyTSmaxShapeHist");
  studyTSmeanShapeHist_ = iConfig.getUntrackedParameter<bool>("studyTSmeanShapeHist");
  studyDiffAmplHist_ = iConfig.getUntrackedParameter<bool>("studyDiffAmplHist");
  studyCalibCellsHist_ = iConfig.getUntrackedParameter<bool>("studyCalibCellsHist");
  studyADCAmplHist_ = iConfig.getUntrackedParameter<bool>("studyADCAmplHist");
  studyPedestalsHist_ = iConfig.getUntrackedParameter<bool>("studyPedestalsHist");
  studyPedestalCorrelations_ = iConfig.getUntrackedParameter<bool>("studyPedestalCorrelations");
  useADCmassive_ = iConfig.getUntrackedParameter<bool>("useADCmassive");
  useADCfC_ = iConfig.getUntrackedParameter<bool>("useADCfC");
  useADCcounts_ = iConfig.getUntrackedParameter<bool>("useADCcounts");
  usePedestalSubtraction_ = iConfig.getUntrackedParameter<bool>("usePedestalSubtraction");
  usecontinuousnumbering_ = iConfig.getUntrackedParameter<bool>("usecontinuousnumbering");
  flagLaserRaddam_ = iConfig.getParameter<int>("flagLaserRaddam");                                          //
  flagIterativeMethodCalibrationGroup_ = iConfig.getParameter<int>("flagIterativeMethodCalibrationGroup");  //
  flagfitshunt1pedorledlowintensity_ = iConfig.getParameter<int>("flagfitshunt1pedorledlowintensity");      //
  flagabortgaprejected_ = iConfig.getParameter<int>("flagabortgaprejected");                                //
  bcnrejectedlow_ = iConfig.getParameter<int>("bcnrejectedlow");                                            //
  bcnrejectedhigh_ = iConfig.getParameter<int>("bcnrejectedhigh");                                          //
  ratioHBMin_ = iConfig.getParameter<double>("ratioHBMin");                                                 //
  ratioHBMax_ = iConfig.getParameter<double>("ratioHBMax");                                                 //
  ratioHEMin_ = iConfig.getParameter<double>("ratioHEMin");                                                 //
  ratioHEMax_ = iConfig.getParameter<double>("ratioHEMax");                                                 //
  ratioHFMin_ = iConfig.getParameter<double>("ratioHFMin");                                                 //
  ratioHFMax_ = iConfig.getParameter<double>("ratioHFMax");                                                 //
  ratioHOMin_ = iConfig.getParameter<double>("ratioHOMin");                                                 //
  ratioHOMax_ = iConfig.getParameter<double>("ratioHOMax");                                                 //
  flagtodefinebadchannel_ = iConfig.getParameter<int>("flagtodefinebadchannel");                            //
  howmanybinsonplots_ = iConfig.getParameter<int>("howmanybinsonplots");                                    //
  splashesUpperLimit_ = iConfig.getParameter<int>("splashesUpperLimit");                                    //
  flagtoaskrunsorls_ = iConfig.getParameter<int>("flagtoaskrunsorls");                                      //
  flagestimatornormalization_ = iConfig.getParameter<int>("flagestimatornormalization");                    //
  flagcpuoptimization_ = iConfig.getParameter<int>("flagcpuoptimization");                                  //
  flagupgradeqie1011_ = iConfig.getParameter<int>("flagupgradeqie1011");                                    //
  flagsipmcorrection_ = iConfig.getParameter<int>("flagsipmcorrection");                                    //
  flaguseshunt_ = iConfig.getParameter<int>("flaguseshunt");                                                //
  lsdep_cut1_peak_HBdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HBdepth1");
  lsdep_cut1_peak_HBdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HBdepth2");
  lsdep_cut1_peak_HEdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth1");
  lsdep_cut1_peak_HEdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth2");
  lsdep_cut1_peak_HEdepth3_ = iConfig.getParameter<int>("lsdep_cut1_peak_HEdepth3");
  lsdep_cut1_peak_HFdepth1_ = iConfig.getParameter<int>("lsdep_cut1_peak_HFdepth1");
  lsdep_cut1_peak_HFdepth2_ = iConfig.getParameter<int>("lsdep_cut1_peak_HFdepth2");
  lsdep_cut1_peak_HOdepth4_ = iConfig.getParameter<int>("lsdep_cut1_peak_HOdepth4");
  lsdep_cut3_max_HBdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HBdepth1");
  lsdep_cut3_max_HBdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HBdepth2");
  lsdep_cut3_max_HEdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth1");
  lsdep_cut3_max_HEdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth2");
  lsdep_cut3_max_HEdepth3_ = iConfig.getParameter<int>("lsdep_cut3_max_HEdepth3");
  lsdep_cut3_max_HFdepth1_ = iConfig.getParameter<int>("lsdep_cut3_max_HFdepth1");
  lsdep_cut3_max_HFdepth2_ = iConfig.getParameter<int>("lsdep_cut3_max_HFdepth2");
  lsdep_cut3_max_HOdepth4_ = iConfig.getParameter<int>("lsdep_cut3_max_HOdepth4");
  lsdep_estimator1_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth1");
  lsdep_estimator1_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth2");
  lsdep_estimator1_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth1");
  lsdep_estimator1_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth2");
  lsdep_estimator1_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth3");
  lsdep_estimator1_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth1");
  lsdep_estimator1_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth2");
  lsdep_estimator1_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HOdepth4");
  lsdep_estimator1_HEdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth4");
  lsdep_estimator1_HEdepth5_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth5");
  lsdep_estimator1_HEdepth6_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth6");
  lsdep_estimator1_HEdepth7_ = iConfig.getParameter<double>("lsdep_estimator1_HEdepth7");
  lsdep_estimator1_HFdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth3");
  lsdep_estimator1_HFdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HFdepth4");
  lsdep_estimator1_HBdepth3_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth3");
  lsdep_estimator1_HBdepth4_ = iConfig.getParameter<double>("lsdep_estimator1_HBdepth4");
  lsdep_estimator2_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HBdepth1");
  lsdep_estimator2_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HBdepth2");
  lsdep_estimator2_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth1");
  lsdep_estimator2_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth2");
  lsdep_estimator2_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator2_HEdepth3");
  lsdep_estimator2_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator2_HFdepth1");
  lsdep_estimator2_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator2_HFdepth2");
  lsdep_estimator2_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator2_HOdepth4");
  lsdep_estimator3_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HBdepth1");
  lsdep_estimator3_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HBdepth2");
  lsdep_estimator3_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth1");
  lsdep_estimator3_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth2");
  lsdep_estimator3_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator3_HEdepth3");
  lsdep_estimator3_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator3_HFdepth1");
  lsdep_estimator3_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator3_HFdepth2");
  lsdep_estimator3_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator3_HOdepth4");
  lsdep_estimator4_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HBdepth1");
  lsdep_estimator4_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HBdepth2");
  lsdep_estimator4_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth1");
  lsdep_estimator4_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth2");
  lsdep_estimator4_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator4_HEdepth3");
  lsdep_estimator4_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator4_HFdepth1");
  lsdep_estimator4_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator4_HFdepth2");
  lsdep_estimator4_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator4_HOdepth4");
  lsdep_estimator5_HBdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HBdepth1");
  lsdep_estimator5_HBdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HBdepth2");
  lsdep_estimator5_HEdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth1");
  lsdep_estimator5_HEdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth2");
  lsdep_estimator5_HEdepth3_ = iConfig.getParameter<double>("lsdep_estimator5_HEdepth3");
  lsdep_estimator5_HFdepth1_ = iConfig.getParameter<double>("lsdep_estimator5_HFdepth1");
  lsdep_estimator5_HFdepth2_ = iConfig.getParameter<double>("lsdep_estimator5_HFdepth2");
  lsdep_estimator5_HOdepth4_ = iConfig.getParameter<double>("lsdep_estimator5_HOdepth4");
  forallestimators_amplitude_bigger_ = iConfig.getParameter<double>("forallestimators_amplitude_bigger");
  rmsHBMin_ = iConfig.getParameter<double>("rmsHBMin");                    //
  rmsHBMax_ = iConfig.getParameter<double>("rmsHBMax");                    //
  rmsHEMin_ = iConfig.getParameter<double>("rmsHEMin");                    //
  rmsHEMax_ = iConfig.getParameter<double>("rmsHEMax");                    //
  rmsHFMin_ = iConfig.getParameter<double>("rmsHFMin");                    //
  rmsHFMax_ = iConfig.getParameter<double>("rmsHFMax");                    //
  rmsHOMin_ = iConfig.getParameter<double>("rmsHOMin");                    //
  rmsHOMax_ = iConfig.getParameter<double>("rmsHOMax");                    //
  ADCAmplHBMin_ = iConfig.getParameter<double>("ADCAmplHBMin");            //
  ADCAmplHEMin_ = iConfig.getParameter<double>("ADCAmplHEMin");            //
  ADCAmplHOMin_ = iConfig.getParameter<double>("ADCAmplHOMin");            //
  ADCAmplHFMin_ = iConfig.getParameter<double>("ADCAmplHFMin");            //
  ADCAmplHBMax_ = iConfig.getParameter<double>("ADCAmplHBMax");            //
  ADCAmplHEMax_ = iConfig.getParameter<double>("ADCAmplHEMax");            //
  ADCAmplHOMax_ = iConfig.getParameter<double>("ADCAmplHOMax");            //
  ADCAmplHFMax_ = iConfig.getParameter<double>("ADCAmplHFMax");            //
  pedestalwHBMax_ = iConfig.getParameter<double>("pedestalwHBMax");        //
  pedestalwHEMax_ = iConfig.getParameter<double>("pedestalwHEMax");        //
  pedestalwHFMax_ = iConfig.getParameter<double>("pedestalwHFMax");        //
  pedestalwHOMax_ = iConfig.getParameter<double>("pedestalwHOMax");        //
  pedestalHBMax_ = iConfig.getParameter<double>("pedestalHBMax");          //
  pedestalHEMax_ = iConfig.getParameter<double>("pedestalHEMax");          //
  pedestalHFMax_ = iConfig.getParameter<double>("pedestalHFMax");          //
  pedestalHOMax_ = iConfig.getParameter<double>("pedestalHOMax");          //
  calibrADCHBMin_ = iConfig.getParameter<double>("calibrADCHBMin");        //
  calibrADCHEMin_ = iConfig.getParameter<double>("calibrADCHEMin");        //
  calibrADCHOMin_ = iConfig.getParameter<double>("calibrADCHOMin");        //
  calibrADCHFMin_ = iConfig.getParameter<double>("calibrADCHFMin");        //
  calibrADCHBMax_ = iConfig.getParameter<double>("calibrADCHBMax");        //
  calibrADCHEMax_ = iConfig.getParameter<double>("calibrADCHEMax");        //
  calibrADCHOMax_ = iConfig.getParameter<double>("calibrADCHOMax");        //
  calibrADCHFMax_ = iConfig.getParameter<double>("calibrADCHFMax");        //
  calibrRatioHBMin_ = iConfig.getParameter<double>("calibrRatioHBMin");    //
  calibrRatioHEMin_ = iConfig.getParameter<double>("calibrRatioHEMin");    //
  calibrRatioHOMin_ = iConfig.getParameter<double>("calibrRatioHOMin");    //
  calibrRatioHFMin_ = iConfig.getParameter<double>("calibrRatioHFMin");    //
  calibrRatioHBMax_ = iConfig.getParameter<double>("calibrRatioHBMax");    //
  calibrRatioHEMax_ = iConfig.getParameter<double>("calibrRatioHEMax");    //
  calibrRatioHOMax_ = iConfig.getParameter<double>("calibrRatioHOMax");    //
  calibrRatioHFMax_ = iConfig.getParameter<double>("calibrRatioHFMax");    //
  calibrTSmaxHBMin_ = iConfig.getParameter<double>("calibrTSmaxHBMin");    //
  calibrTSmaxHEMin_ = iConfig.getParameter<double>("calibrTSmaxHEMin");    //
  calibrTSmaxHOMin_ = iConfig.getParameter<double>("calibrTSmaxHOMin");    //
  calibrTSmaxHFMin_ = iConfig.getParameter<double>("calibrTSmaxHFMin");    //
  calibrTSmaxHBMax_ = iConfig.getParameter<double>("calibrTSmaxHBMax");    //
  calibrTSmaxHEMax_ = iConfig.getParameter<double>("calibrTSmaxHEMax");    //
  calibrTSmaxHOMax_ = iConfig.getParameter<double>("calibrTSmaxHOMax");    //
  calibrTSmaxHFMax_ = iConfig.getParameter<double>("calibrTSmaxHFMax");    //
  calibrTSmeanHBMin_ = iConfig.getParameter<double>("calibrTSmeanHBMin");  //
  calibrTSmeanHEMin_ = iConfig.getParameter<double>("calibrTSmeanHEMin");  //
  calibrTSmeanHOMin_ = iConfig.getParameter<double>("calibrTSmeanHOMin");  //
  calibrTSmeanHFMin_ = iConfig.getParameter<double>("calibrTSmeanHFMin");  //
  calibrTSmeanHBMax_ = iConfig.getParameter<double>("calibrTSmeanHBMax");  //
  calibrTSmeanHEMax_ = iConfig.getParameter<double>("calibrTSmeanHEMax");  //
  calibrTSmeanHOMax_ = iConfig.getParameter<double>("calibrTSmeanHOMax");  //
  calibrTSmeanHFMax_ = iConfig.getParameter<double>("calibrTSmeanHFMax");  //
  calibrWidthHBMin_ = iConfig.getParameter<double>("calibrWidthHBMin");    //
  calibrWidthHEMin_ = iConfig.getParameter<double>("calibrWidthHEMin");    //
  calibrWidthHOMin_ = iConfig.getParameter<double>("calibrWidthHOMin");    //
  calibrWidthHFMin_ = iConfig.getParameter<double>("calibrWidthHFMin");    //
  calibrWidthHBMax_ = iConfig.getParameter<double>("calibrWidthHBMax");    //
  calibrWidthHEMax_ = iConfig.getParameter<double>("calibrWidthHEMax");    //
  calibrWidthHOMax_ = iConfig.getParameter<double>("calibrWidthHOMax");    //
  calibrWidthHFMax_ = iConfig.getParameter<double>("calibrWidthHFMax");    //
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile");
  MAPOutputFileName = iConfig.getUntrackedParameter<std::string>("MAPOutFile");
  TSpeakHBMin_ = iConfig.getParameter<double>("TSpeakHBMin");  //
  TSpeakHBMax_ = iConfig.getParameter<double>("TSpeakHBMax");  //
  TSpeakHEMin_ = iConfig.getParameter<double>("TSpeakHEMin");  //
  TSpeakHEMax_ = iConfig.getParameter<double>("TSpeakHEMax");  //
  TSpeakHFMin_ = iConfig.getParameter<double>("TSpeakHFMin");  //
  TSpeakHFMax_ = iConfig.getParameter<double>("TSpeakHFMax");  //
  TSpeakHOMin_ = iConfig.getParameter<double>("TSpeakHOMin");  //
  TSpeakHOMax_ = iConfig.getParameter<double>("TSpeakHOMax");  //
  TSmeanHBMin_ = iConfig.getParameter<double>("TSmeanHBMin");  //
  TSmeanHBMax_ = iConfig.getParameter<double>("TSmeanHBMax");  //
  TSmeanHEMin_ = iConfig.getParameter<double>("TSmeanHEMin");  //
  TSmeanHEMax_ = iConfig.getParameter<double>("TSmeanHEMax");  //
  TSmeanHFMin_ = iConfig.getParameter<double>("TSmeanHFMin");  //
  TSmeanHFMax_ = iConfig.getParameter<double>("TSmeanHFMax");  //
  TSmeanHOMin_ = iConfig.getParameter<double>("TSmeanHOMin");  //
  TSmeanHOMax_ = iConfig.getParameter<double>("TSmeanHOMax");  //
  lsmin_ = iConfig.getParameter<int>("lsmin");                 //
  lsmax_ = iConfig.getParameter<int>("lsmax");                 //
  alsmin = lsmin_;
  blsmax = lsmax_;
  nlsminmax = lsmax_ - lsmin_ + 1;
  numOfLaserEv = 0;
  local_event = 0;
  numOfTS = 10;
  run0 = -1;
  runcounter = 0;
  eventcounter = 0;
  lumi = 0;
  ls0 = -1;
  lscounter = 0;
  lscounterM1 = 0;
  lscounter10 = 0;
  nevcounter = 0;
  lscounterrun = 0;
  lscounterrun10 = 0;
  nevcounter0 = 0;
  nevcounter00 = 0;
  for (int k0 = 0; k0 < nsub; k0++) {
    for (int k1 = 0; k1 < ndepth; k1++) {
      for (int k2 = 0; k2 < neta; k2++) {
        if (k0 == 1) {
          mapRADDAM_HED2[k1][k2] = 0.;
          mapRADDAM_HED20[k1][k2] = 0.;
        }
        for (int k3 = 0; k3 < nphi; k3++) {
          sumEstimator0[k0][k1][k2][k3] = 0.;
          sumEstimator1[k0][k1][k2][k3] = 0.;
          sumEstimator2[k0][k1][k2][k3] = 0.;
          sumEstimator3[k0][k1][k2][k3] = 0.;
          sumEstimator4[k0][k1][k2][k3] = 0.;
          sumEstimator5[k0][k1][k2][k3] = 0.;
          sumEstimator6[k0][k1][k2][k3] = 0.;
          sum0Estimator[k0][k1][k2][k3] = 0.;
          if (k0 == 1) {
            mapRADDAM_HE[k1][k2][k3] = 0.;
            mapRADDAM0_HE[k1][k2][k3] = 0;
          }
        }  //for
      }    //for
    }      //for
  }        //for
  averSIGNALoccupancy_HB = 0.;
  averSIGNALoccupancy_HE = 0.;
  averSIGNALoccupancy_HF = 0.;
  averSIGNALoccupancy_HO = 0.;
  averSIGNALsumamplitude_HB = 0.;
  averSIGNALsumamplitude_HE = 0.;
  averSIGNALsumamplitude_HF = 0.;
  averSIGNALsumamplitude_HO = 0.;
  averNOSIGNALoccupancy_HB = 0.;
  averNOSIGNALoccupancy_HE = 0.;
  averNOSIGNALoccupancy_HF = 0.;
  averNOSIGNALoccupancy_HO = 0.;
  averNOSIGNALsumamplitude_HB = 0.;
  averNOSIGNALsumamplitude_HE = 0.;
  averNOSIGNALsumamplitude_HF = 0.;
  averNOSIGNALsumamplitude_HO = 0.;
  maxxSUM1 = 0.;
  maxxSUM2 = 0.;
  maxxSUM3 = 0.;
  maxxSUM4 = 0.;
  maxxOCCUP1 = 0.;
  maxxOCCUP2 = 0.;
  maxxOCCUP3 = 0.;
  maxxOCCUP4 = 0.;
  testmetka = 0;
}
CMTRawAnalyzer::~CMTRawAnalyzer() {}
void CMTRawAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  iSetup.get<HcalDbRecord>().get(conditions);
  iSetup.get<HcalRecNumberingRecord>().get(topo_);
  if (MAPcreation > 0) {
    topo = &*topo_;
    if (flagupgradeqie1011_ == 1)
      fillMAP();
    MAPcreation = 0;
  }
  nevent++;
  nevent50 = nevent / 50;
  Run = iEvent.id().run();
  Nevent = iEvent.id().event();     // event number = global_event
  lumi = iEvent.luminosityBlock();  // lumi section
  bcn = iEvent.bunchCrossing();
  orbitNum = iEvent.orbitNumber();
  int outabortgap = 1;
  if (bcn >= bcnrejectedlow_ && bcn <= bcnrejectedhigh_)
    outabortgap = 0;  //  if(bcn>=3446 && bcn<=3564)

  if ((flagabortgaprejected_ == 1 && outabortgap == 1) || (flagabortgaprejected_ == 0 && outabortgap == 0) ||
      flagabortgaprejected_ == 2) {
    if (run0 != Run) {
      ++runcounter;
      if (runcounter != 1) {
        nevcounter00 = eventcounter;
        cout << " --------------------------------------- " << endl;
        cout << " for Run = " << run0 << " with runcounter = " << runcounter - 1 << " #ev = " << eventcounter << endl;
        cout << " #LS =  " << lscounterrun << " #LS10 =  " << lscounterrun10 << " Last LS =  " << ls0 << endl;
        cout << " --------------------------------------------- " << endl;
        h_nls_per_run->Fill(float(lscounterrun));
        h_nls_per_run10->Fill(float(lscounterrun10));
        lscounterrun = 0;
        lscounterrun10 = 0;
      }  // runcounter > 1
      cout << " ---------***********************------------- " << endl;
      cout << " New Run =  " << Run << " runcounter =  " << runcounter << endl;
      cout << " ------- " << endl;
      run0 = Run;
      eventcounter = 0;
      ls0 = -1;
    }  // new run
    else {
      nevcounter00 = 0;
    }  //else new run
    ++eventcounter;
    if (ls0 != lumi) {
      if (ls0 != -1) {
        h_nevents_per_eachLS->Fill(float(lscounter), float(nevcounter));  //
        nevcounter0 = nevcounter;
      }  // ls0>-1
      lscounter++;
      lscounterrun++;
      if (usecontinuousnumbering_) {
        lscounterM1 = lscounter - 1;
      } else {
        lscounterM1 = ls0;
      }
      if (ls0 != -1)
        h_nevents_per_eachRealLS->Fill(float(lscounterM1), float(nevcounter));  //
      h_lsnumber_per_eachLS->Fill(float(lscounter), float(lumi));
      if (nevcounter > 10.) {
        ++lscounter10;
        ++lscounterrun10;
      }
      h_nevents_per_LS->Fill(float(nevcounter));
      h_nevents_per_LSzoom->Fill(float(nevcounter));
      nevcounter = 0;
      ls0 = lumi;
    }  // new lumi
    else {
      nevcounter0 = 0;
    }              //else new lumi
    ++nevcounter;  // #ev in LS
                   //////
    if (flagtoaskrunsorls_ == 0) {
      lscounterM1 = runcounter;
      nevcounter0 = nevcounter00;
    }
    if (nevcounter0 != 0 || nevcounter > 99999) {
      if (nevcounter > 99999)
        nevcounter0 = 1;
      ///////  int sub= cell.subdet();  1-HB, 2-HE, 3-HO, 4-HF
      ////////////            k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      ////////////         k1(depth-1): = 0 - 3 or depth: = 1 - 4;
      unsigned long int pcountall1 = 0;
      unsigned long int pcountall3 = 0;
      unsigned long int pcountall6 = 0;
      unsigned long int pcountall8 = 0;
      int pcountmin1 = 0;
      int pcountmin3 = 0;
      int pcountmin6 = 0;
      int pcountmin8 = 0;
      unsigned long int mcountall1 = 0;
      unsigned long int mcountall3 = 0;
      unsigned long int mcountall6 = 0;
      unsigned long int mcountall8 = 0;
      int mcountmin1 = 0;
      int mcountmin3 = 0;
      int mcountmin6 = 0;
      int mcountmin8 = 0;
      int pnnbins1 = 0;
      int pnnbins3 = 0;
      int pnnbins6 = 0;
      int pnnbins8 = 0;
      int pnnmin1 = 999999999;
      int pnnmin3 = 999999999;
      int pnnmin6 = 999999999;
      int pnnmin8 = 999999999;
      int mnnbins1 = 0;
      int mnnbins3 = 0;
      int mnnbins6 = 0;
      int mnnbins8 = 0;
      int mnnmin1 = 999999999;
      int mnnmin3 = 999999999;
      int mnnmin6 = 999999999;
      int mnnmin8 = 999999999;
      for (int k0 = 0; k0 < nsub; k0++) {
        for (int k1 = 0; k1 < ndepth; k1++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            for (int k2 = 0; k2 < neta; k2++) {
              int ieta = k2 - 41;
              // ------------------------------------------------------------sumEstimator0
              if (sumEstimator0[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator0[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator0[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator0[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS1->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS1->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS2->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS3->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS4->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumPedestalLS5->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumPedestalLS6->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumPedestalLS7->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumPedestalLS8->Fill(bbbc / bbb1);
                    h_2DsumPedestalLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumPedestalLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumPedestalperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0PedestalperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator0[k0][k1][k2][k3] != 0.

              // ---------------------------------------------------------------------------------------------------------------------------sumEstimator1
              if (sumEstimator1[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator1[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator1[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator1[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }
                //flag for ask type of Normalization for CMT estimators:
                //=0-normalizationOn#evOfLS;   =1-averagedMeanChannelVariable;   =2-averageVariable-normalizationOn#entriesInLS;
                //flagestimatornormalization = cms.int32(2), !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                // zhokin 18.10.2018 STUDY:               CALL  HFF2 (ID,NID,X,Y,W)
                if (lscounterM1 >= lsmin_ && lscounterM1 < lsmax_) {
                  //                                       INDEXIES:
                  int kkkk2 = (k2 - 1) / 4;
                  if (k2 == 0)
                    kkkk2 = 1.;
                  else
                    kkkk2 += 2;              //kkkk2= 1-22
                  int kkkk3 = (k3) / 4 + 1;  //kkkk3= 1-18
                  //                                       PACKING
                  //kkkk2= 1-22 ;kkkk3= 1-18
                  int ietaphi = 0;
                  ietaphi = ((kkkk2)-1) * znphi + (kkkk3);
                  //  Outout is       ietaphi = 1 - 396 ( # =396; in histo,booking is: 1 - 397 )

                  double bbb3 = 0.;
                  if (bbb1 != 0.)
                    bbb3 = bbbc / bbb1;
                  // very very wrong if below:
                  //		if(bbb3 != 0.) {

                  if (k0 == 0) {
                    h_2DsumADCAmplEtaPhiLs0->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HB
                    h_2DsumADCAmplEtaPhiLs00->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HB
                  }
                  if (k0 == 1) {
                    h_2DsumADCAmplEtaPhiLs1->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HE
                    h_2DsumADCAmplEtaPhiLs10->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HE
                  }
                  if (k0 == 2) {
                    h_2DsumADCAmplEtaPhiLs2->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HO
                    h_2DsumADCAmplEtaPhiLs20->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HO
                  }
                  if (k0 == 3) {
                    h_2DsumADCAmplEtaPhiLs3->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HF
                    h_2DsumADCAmplEtaPhiLs30->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HF
                  }

                  h_sumADCAmplEtaPhiLs->Fill(bbb3);
                  h_sumADCAmplEtaPhiLs_bbbc->Fill(bbbc);
                  h_sumADCAmplEtaPhiLs_bbb1->Fill(bbb1);
                  h_sumADCAmplEtaPhiLs_lscounterM1orbitNum->Fill(float(lscounterM1), float(orbitNum));
                  h_sumADCAmplEtaPhiLs_orbitNum->Fill(float(orbitNum), 1.);
                  h_sumADCAmplEtaPhiLs_lscounterM1->Fill(float(lscounterM1), 1.);
                  h_sumADCAmplEtaPhiLs_ietaphi->Fill(float(ietaphi));

                  //		}// bb3
                }  // lscounterM1 >= lsmin_ && lscounterM1 < lsmax_

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS1copy1->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy2->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy3->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy4->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1copy5->Fill(bbbc / bbb1);
                    h_sumADCAmplLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                      h_2DsumADCAmplLS1->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth1_)
                      h_2DsumADCAmplLS1_LSselected->Fill(double(ieta), double(k3), bbbc);

                    h_2D0sumADCAmplLS1->Fill(double(ieta), double(k3), bbb1);

                    h_sumADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                      h_sumCutADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS1->Fill(float(lscounterM1), bbb1);

                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 25.) {
                        pcountall1 += bbb1;
                        pcountmin1 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 25.) {
                        mcountall1 += bbb1;
                        mcountmin1 += bbb1;
                      }
                      //////////////////////////////
                    }
                  }
                  // HBdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                      h_2DsumADCAmplLS2->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth2_)
                      h_2DsumADCAmplLS2_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                      h_sumCutADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS2->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS1_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS1_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS1_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }
                  // HBdepth3 upgrade
                  if (k1 + 1 == 3) {
                    h_sumADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                      h_sumCutADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                      h_2DsumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==3)

                  // HBdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                      h_sumCutADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                      h_2DsumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)
                }

                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                      h_2DsumADCAmplLS3->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth1_)
                      h_2DsumADCAmplLS3_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                      h_sumCutADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS3->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 15. && k3 % 2 == 0) {
                        pcountall3 += bbb1;
                        pcountmin3 += bbb1;
                      }
                      //////////////////////////////
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 15. && k3 % 2 == 0) {
                        mcountall3 += bbb1;
                        mcountmin3 += bbb1;
                      }
                      //////////////////////////////
                    }
                  }
                  // HEdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                      h_2DsumADCAmplLS4->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth2_)
                      h_2DsumADCAmplLS4_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                      h_sumCutADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS4->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }
                  // HEdepth3
                  if (k1 + 1 == 3) {
                    h_sumADCAmplLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                      h_2DsumADCAmplLS5->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth3_)
                      h_2DsumADCAmplLS5_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                      h_sumCutADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS5->Fill(float(lscounterM1), bbb1);
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS3_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS3_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS3_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }  //if(k1+1  ==3
                  // HEdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                      h_sumCutADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                      h_2DsumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)

                  // HEdepth5 upgrade
                  if (k1 + 1 == 5) {
                    h_sumADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                      h_sumCutADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                      h_2DsumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==5)

                  // HEdepth6 upgrade
                  if (k1 + 1 == 6) {
                    h_sumADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                      h_sumCutADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                      h_2DsumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==6)

                  // HEdepth7 upgrade
                  if (k1 + 1 == 7) {
                    h_sumADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                      h_sumCutADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                      h_2DsumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==7)
                }    //if(k0==1) =HE
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumADCAmplLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                      h_2DsumADCAmplLS6->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth1_)
                      h_2DsumADCAmplLS6_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                      h_sumCutADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS6->Fill(float(lscounterM1), bbb1);

                    ///////////////////////////////////////////////////////// error-A
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 20.) {
                        pcountall6 += bbb1;
                        pcountmin6 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 20.) {
                        mcountall6 += bbb1;
                        mcountmin6 += bbb1;
                      }
                      //////////////////////////////
                    }
                    /////////////////////////////////////////////////////////
                  }  //if(k1+1  ==1)

                  // HFdepth2
                  if (k1 + 1 == 2) {
                    h_sumADCAmplLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                      h_2DsumADCAmplLS7->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth2_)
                      h_2DsumADCAmplLS7_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                      h_sumCutADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS7->Fill(float(lscounterM1), bbb1);

                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_P2->Fill(float(lscounterM1), bbb1);
                      }
                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS6_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS6_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS6_M2->Fill(float(lscounterM1), bbb1);
                      }
                    }
                  }  //if(k1+1  ==2)

                  // HFdepth3 upgrade
                  if (k1 + 1 == 3) {
                    h_sumADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                      h_sumCutADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS6u->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                      h_2DsumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==3)

                  // HFdepth4 upgrade
                  if (k1 + 1 == 4) {
                    h_sumADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                      h_sumCutADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS7u->Fill(float(lscounterM1), bbb1);

                    if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                      h_2DsumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbb1);
                  }  //if(k1+1  ==4)

                }  //end HF

                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumADCAmplLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                      h_2DsumADCAmplLS8->Fill(double(ieta), double(k3), bbbc);
                    if (bbbc / bbb1 > 2. * lsdep_estimator1_HOdepth4_)
                      h_2DsumADCAmplLS8_LSselected->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumADCAmplLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                      h_sumCutADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0ADCAmplperLS8->Fill(float(lscounterM1), bbb1);

                    ///////////////////////////////////////////////////////// error-A
                    if (ieta > 0) {
                      if (k3 < 36) {
                        h_sumADCAmplperLS8_P1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_P1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS8_P2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_P2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// P
                      if (bbbc / bbb1 > 80.) {
                        pcountall8 += bbb1;
                        pcountmin8 += bbb1;
                      }
                      //////////////////////////////

                    } else {
                      if (k3 < 36) {
                        h_sumADCAmplperLS8_M1->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_M1->Fill(float(lscounterM1), bbb1);
                      } else {
                        h_sumADCAmplperLS8_M2->Fill(float(lscounterM1), bbbc);
                        h_sum0ADCAmplperLS8_M2->Fill(float(lscounterM1), bbb1);
                      }
                      ////////////////////////////// M
                      if (bbbc / bbb1 > 80.) {
                        mcountall8 += bbb1;
                        mcountmin8 += bbb1;
                      }
                      //////////////////////////////
                    }
                    /////////////////////////////////////////////////////////
                  }
                }
              }  //if(sumEstimator1[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator2
              if (sumEstimator2[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator2[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator2[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator2[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                      h_2DsumTSmeanALS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                      h_sumCutTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator2_HBdepth1_)
                      h_sumTSmeanAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                      h_2DsumTSmeanALS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                      h_sumCutTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                      h_2DsumTSmeanALS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                      h_sumCutTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                      h_2DsumTSmeanALS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                      h_sumCutTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumTSmeanALS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                      h_2DsumTSmeanALS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                      h_sumCutTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmeanALS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                      h_2DsumTSmeanALS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                      h_sumCutTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmeanALS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                      h_2DsumTSmeanALS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                      h_sumCutTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumTSmeanALS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                      h_2DsumTSmeanALS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmeanALS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                      h_sumCutTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmeanAperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator2[k0][k1][k2][k3] != 0.

              // ------------------------------------------------------------------------------------------------------------------------sumEstimator3
              if (sumEstimator3[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator3[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator3[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator3[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                      h_2DsumTSmaxALS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                      h_sumCutTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator3_HBdepth1_)
                      h_sumTSmaxAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                      h_2DsumTSmaxALS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                      h_sumCutTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                      h_2DsumTSmaxALS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                      h_sumCutTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                      h_2DsumTSmaxALS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                      h_sumCutTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumTSmaxALS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                      h_2DsumTSmaxALS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                      h_sumCutTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumTSmaxALS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                      h_2DsumTSmaxALS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                      h_sumCutTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumTSmaxALS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                      h_2DsumTSmaxALS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                      h_sumCutTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumTSmaxALS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                      h_2DsumTSmaxALS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumTSmaxALS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                      h_sumCutTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0TSmaxAperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator3[k0][k1][k2][k3] != 0.

              // ------------------------------------------------------------------------------------------------------------------------sumEstimator4
              if (sumEstimator4[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator4[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator4[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator4[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                      h_2DsumAmplitudeLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                      h_sumCutAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator4_HBdepth1_)
                      h_sumAmplitudeperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                      h_2DsumAmplitudeLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                      h_sumCutAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                      h_2DsumAmplitudeLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                      h_sumCutAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                      h_2DsumAmplitudeLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                      h_sumCutAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumAmplitudeLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                      h_2DsumAmplitudeLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                      h_sumCutAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplitudeLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                      h_2DsumAmplitudeLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                      h_sumCutAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplitudeLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                      h_2DsumAmplitudeLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                      h_sumCutAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HBdepth1
                  if (k1 + 1 == 4) {
                    h_sumAmplitudeLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                      h_2DsumAmplitudeLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplitudeLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                      h_sumCutAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplitudeperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator4[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator5
              if (sumEstimator5[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator5[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator5[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator5[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS1->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                      h_2DsumAmplLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS1->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                      h_sumCutAmplperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS1->Fill(float(lscounterM1), bbb1);
                    if (bbbc / bbb1 > 2. * lsdep_estimator5_HBdepth1_)
                      h_sumAmplperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS2->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                      h_2DsumAmplLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS2->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                      h_sumCutAmplperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS3->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                      h_2DsumAmplLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS3->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                      h_sumCutAmplperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS4->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                      h_2DsumAmplLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS4->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                      h_sumCutAmplperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumAmplLS5->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                      h_2DsumAmplLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS5->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                      h_sumCutAmplperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumAmplLS6->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                      h_2DsumAmplLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS6->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                      h_sumCutAmplperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumAmplLS7->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                      h_2DsumAmplLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS7->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                      h_sumCutAmplperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumAmplLS8->Fill(bbbc / bbb1);
                    if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                      h_2DsumAmplLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumAmplLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumAmplperLS8->Fill(float(lscounterM1), bbbc);
                    if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                      h_sumCutAmplperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0AmplperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
              }  //if(sumEstimator5[k0][k1][k2][k3] != 0.
              // ------------------------------------------------------------------------------------------------------------------------sumEstimator6 (Error-B)
              if (sumEstimator6[k0][k1][k2][k3] != 0.) {
                // fill histoes:
                double bbbc = 0.;
                if (flagestimatornormalization_ == 0)
                  bbbc = sumEstimator6[k0][k1][k2][k3] / nevcounter0;
                if (flagestimatornormalization_ == 1)
                  bbbc = sumEstimator6[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
                double bbb1 = 1.;
                if (flagestimatornormalization_ == 2) {
                  bbbc = sumEstimator6[k0][k1][k2][k3];
                  bbb1 = sum0Estimator[k0][k1][k2][k3];
                }

                // HB:
                if (k0 == 0) {
                  // HBdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS1->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS1->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS1->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS1->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS1->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS2->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS2->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS2->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS2->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS2->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HE:
                if (k0 == 1) {
                  // HEdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS3->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS3->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS3->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS3->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS3->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS4->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS4->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS4->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS4->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS4->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 3) {
                    h_sumErrorBLS5->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS5->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS5->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS5->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS5->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HF:
                if (k0 == 3) {
                  // HFdepth1
                  if (k1 + 1 == 1) {
                    h_sumErrorBLS6->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS6->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS6->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS6->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS6->Fill(float(lscounterM1), bbb1);
                  }
                  if (k1 + 1 == 2) {
                    h_sumErrorBLS7->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS7->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS7->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS7->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS7->Fill(float(lscounterM1), bbb1);
                  }
                }
                // HO:
                if (k0 == 2) {
                  // HOdepth4
                  if (k1 + 1 == 4) {
                    h_sumErrorBLS8->Fill(bbbc / bbb1);
                    h_2DsumErrorBLS8->Fill(double(ieta), double(k3), bbbc);
                    h_2D0sumErrorBLS8->Fill(double(ieta), double(k3), bbb1);
                    h_sumErrorBperLS8->Fill(float(lscounterM1), bbbc);
                    h_sum0ErrorBperLS8->Fill(float(lscounterM1), bbb1);
                  }
                }
                ///
              }  //if(sumEstimator6[k0][k1][k2][k3] != 0.
            }    //for k2
            // occupancy distributions for error-A:
            // HB
            if (k0 == 0 && k1 == 0) {
              if (pcountmin1 > 0) {
                if (pcountmin1 < pnnmin1)
                  pnnmin1 = pcountmin1;
                pcountmin1 = 0;
                pnnbins1++;
              }
              if (mcountmin1 > 0) {
                if (mcountmin1 < mnnmin1)
                  mnnmin1 = mcountmin1;
                mcountmin1 = 0;
                mnnbins1++;
              }
            }  //
            // HE
            if (k0 == 1 && k1 == 0) {
              if (pcountmin3 > 0) {
                if (pcountmin3 < pnnmin3)
                  pnnmin3 = pcountmin3;
                pcountmin3 = 0;
                pnnbins3++;
              }
              if (mcountmin3 > 0) {
                if (mcountmin3 < mnnmin3)
                  mnnmin3 = mcountmin3;
                mcountmin3 = 0;
                mnnbins3++;
              }
            }  //
            // HO
            if (k0 == 2 && k1 == 3) {
              if (pcountmin8 > 0) {
                if (pcountmin8 < pnnmin8)
                  pnnmin8 = pcountmin8;
                pcountmin8 = 0;
                pnnbins8++;
              }
              if (mcountmin8 > 0) {
                if (mcountmin8 < mnnmin8)
                  mnnmin8 = mcountmin8;
                mcountmin8 = 0;
                mnnbins8++;
              }
            }  //
            // HF
            if (k0 == 3 && k1 == 0) {
              if (pcountmin6 > 0) {
                if (pcountmin6 < pnnmin6)
                  pnnmin6 = pcountmin6;
                pcountmin6 = 0;
                pnnbins6++;
              }
              if (mcountmin6 > 0) {
                if (mcountmin6 < mnnmin6)
                  mnnmin6 = mcountmin6;
                mcountmin6 = 0;
                mnnbins6++;
              }
            }  //

          }  //for k3
        }    //for k1
      }      //for k0
      ///////  int sub= cell.subdet();  1-HB, 2-HE, 3-HO, 4-HF
      ////////////            k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      ////////////         k1(depth-1): = 0 - 3 or depth: = 1 - 4;

      //   cout<<"=============================== lscounterM1 = "<<   (float)lscounterM1    <<endl;

      float patiooccupancy1 = 0.;
      if (pcountall1 != 0)
        patiooccupancy1 = (float)pnnmin1 * mnnbins1 / pcountall1;
      h_RatioOccupancy_HBM->Fill(float(lscounterM1), patiooccupancy1);
      float matiooccupancy1 = 0.;
      if (mcountall1 != 0)
        matiooccupancy1 = (float)mnnmin1 * mnnbins1 / mcountall1;
      h_RatioOccupancy_HBP->Fill(float(lscounterM1), matiooccupancy1);

      float patiooccupancy3 = 0.;
      if (pcountall3 != 0)
        patiooccupancy3 = (float)pnnmin3 * mnnbins3 / pcountall3;
      h_RatioOccupancy_HEM->Fill(float(lscounterM1), patiooccupancy3);
      float matiooccupancy3 = 0.;
      if (mcountall3 != 0)
        matiooccupancy3 = (float)mnnmin3 * mnnbins3 / mcountall3;
      h_RatioOccupancy_HEP->Fill(float(lscounterM1), matiooccupancy3);

      float patiooccupancy6 = 0.;
      if (pcountall6 != 0)
        patiooccupancy6 = (float)pnnmin6 * mnnbins6 / pcountall6;
      h_RatioOccupancy_HFM->Fill(float(lscounterM1), patiooccupancy6);
      float matiooccupancy6 = 0.;
      if (mcountall6 != 0)
        matiooccupancy6 = (float)mnnmin6 * mnnbins6 / mcountall6;
      h_RatioOccupancy_HFP->Fill(float(lscounterM1), matiooccupancy6);

      float patiooccupancy8 = 0.;
      if (pcountall8 != 0)
        patiooccupancy8 = (float)pnnmin8 * mnnbins8 / pcountall8;
      h_RatioOccupancy_HOM->Fill(float(lscounterM1), patiooccupancy8);
      float matiooccupancy8 = 0.;
      if (mcountall8 != 0)
        matiooccupancy8 = (float)mnnmin8 * mnnbins8 / mcountall8;
      h_RatioOccupancy_HOP->Fill(float(lscounterM1), matiooccupancy8);

      for (int k0 = 0; k0 < nsub; k0++) {
        for (int k1 = 0; k1 < ndepth; k1++) {
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              // reset massives:
              sumEstimator0[k0][k1][k2][k3] = 0.;
              sumEstimator1[k0][k1][k2][k3] = 0.;
              sumEstimator2[k0][k1][k2][k3] = 0.;
              sumEstimator3[k0][k1][k2][k3] = 0.;
              sumEstimator4[k0][k1][k2][k3] = 0.;
              sumEstimator5[k0][k1][k2][k3] = 0.;
              sumEstimator6[k0][k1][k2][k3] = 0.;
              sum0Estimator[k0][k1][k2][k3] = 0.;
            }  //for
          }    //for
        }      //for
      }        //for

      //------------------------------------------------------                        averSIGNAL
      averSIGNALoccupancy_HB /= float(nevcounter0);
      h_averSIGNALoccupancy_HB->Fill(float(lscounterM1), averSIGNALoccupancy_HB);
      averSIGNALoccupancy_HE /= float(nevcounter0);
      h_averSIGNALoccupancy_HE->Fill(float(lscounterM1), averSIGNALoccupancy_HE);
      averSIGNALoccupancy_HF /= float(nevcounter0);
      h_averSIGNALoccupancy_HF->Fill(float(lscounterM1), averSIGNALoccupancy_HF);
      averSIGNALoccupancy_HO /= float(nevcounter0);
      h_averSIGNALoccupancy_HO->Fill(float(lscounterM1), averSIGNALoccupancy_HO);

      averSIGNALoccupancy_HB = 0.;
      averSIGNALoccupancy_HE = 0.;
      averSIGNALoccupancy_HF = 0.;
      averSIGNALoccupancy_HO = 0.;

      //------------------------------------------------------
      averSIGNALsumamplitude_HB /= float(nevcounter0);
      h_averSIGNALsumamplitude_HB->Fill(float(lscounterM1), averSIGNALsumamplitude_HB);
      averSIGNALsumamplitude_HE /= float(nevcounter0);
      h_averSIGNALsumamplitude_HE->Fill(float(lscounterM1), averSIGNALsumamplitude_HE);
      averSIGNALsumamplitude_HF /= float(nevcounter0);
      h_averSIGNALsumamplitude_HF->Fill(float(lscounterM1), averSIGNALsumamplitude_HF);
      averSIGNALsumamplitude_HO /= float(nevcounter0);
      h_averSIGNALsumamplitude_HO->Fill(float(lscounterM1), averSIGNALsumamplitude_HO);

      averSIGNALsumamplitude_HB = 0.;
      averSIGNALsumamplitude_HE = 0.;
      averSIGNALsumamplitude_HF = 0.;
      averSIGNALsumamplitude_HO = 0.;

      //------------------------------------------------------                        averNOSIGNAL
      averNOSIGNALoccupancy_HB /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HB->Fill(float(lscounterM1), averNOSIGNALoccupancy_HB);
      averNOSIGNALoccupancy_HE /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HE->Fill(float(lscounterM1), averNOSIGNALoccupancy_HE);
      averNOSIGNALoccupancy_HF /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HF->Fill(float(lscounterM1), averNOSIGNALoccupancy_HF);
      averNOSIGNALoccupancy_HO /= float(nevcounter0);
      h_averNOSIGNALoccupancy_HO->Fill(float(lscounterM1), averNOSIGNALoccupancy_HO);

      averNOSIGNALoccupancy_HB = 0.;
      averNOSIGNALoccupancy_HE = 0.;
      averNOSIGNALoccupancy_HF = 0.;
      averNOSIGNALoccupancy_HO = 0.;

      //------------------------------------------------------
      averNOSIGNALsumamplitude_HB /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HB->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HB);
      averNOSIGNALsumamplitude_HE /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HE->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HE);
      averNOSIGNALsumamplitude_HF /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HF->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HF);
      averNOSIGNALsumamplitude_HO /= float(nevcounter0);
      h_averNOSIGNALsumamplitude_HO->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HO);

      averNOSIGNALsumamplitude_HB = 0.;
      averNOSIGNALsumamplitude_HE = 0.;
      averNOSIGNALsumamplitude_HF = 0.;
      averNOSIGNALsumamplitude_HO = 0.;

      //------------------------------------------------------   maxxSA and maxxOccupancy
      h_maxxSUMAmpl_HB->Fill(float(lscounterM1), maxxSUM1);
      h_maxxSUMAmpl_HE->Fill(float(lscounterM1), maxxSUM2);
      h_maxxSUMAmpl_HO->Fill(float(lscounterM1), maxxSUM3);
      h_maxxSUMAmpl_HF->Fill(float(lscounterM1), maxxSUM4);
      maxxSUM1 = 0.;
      maxxSUM2 = 0.;
      maxxSUM3 = 0.;
      maxxSUM4 = 0.;
      //------------------------------------------------------
      h_maxxOCCUP_HB->Fill(float(lscounterM1), maxxOCCUP1);
      h_maxxOCCUP_HE->Fill(float(lscounterM1), maxxOCCUP2);
      h_maxxOCCUP_HO->Fill(float(lscounterM1), maxxOCCUP3);
      h_maxxOCCUP_HF->Fill(float(lscounterM1), maxxOCCUP4);
      maxxOCCUP1 = 0.;
      maxxOCCUP2 = 0.;
      maxxOCCUP3 = 0.;
      maxxOCCUP4 = 0.;

      //------------------------------------------------------
    }  //if(nevcounter0 != 0)
       //  POINT1

    /////////////////////////////////////////////////// over DigiCollections:
    // for upgrade:
    for (int k1 = 0; k1 < ndepth; k1++) {
      for (int k2 = 0; k2 < neta; k2++) {
        for (int k3 = 0; k3 < nphi; k3++) {
          if (studyCalibCellsHist_) {
            signal[k1][k2][k3] = 0.;
            calibt[k1][k2][k3] = 0.;
            calibcapiderror[k1][k2][k3] = 0;
            caliba[k1][k2][k3] = 0.;
            calibw[k1][k2][k3] = 0.;
            calib0[k1][k2][k3] = 0.;
            signal3[k1][k2][k3] = 0.;
            calib3[k1][k2][k3] = 0.;
            calib2[k1][k2][k3] = 0.;
          }
          if (studyRunDependenceHist_) {
            for (int k0 = 0; k0 < nsub; k0++) {
              badchannels[k0][k1][k2][k3] = 0;
            }  //for
          }    //if

        }  //for
      }    //for
    }      //for
    for (int k0 = 0; k0 < nsub; k0++) {
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            amplitudechannel[k0][k1][k2][k3] = 0.;
            tocamplchannel[k0][k1][k2][k3] = 0.;
            maprphinorm[k0][k1][k2][k3] = 0.;
          }  //k3
        }    //k2
      }      //k1
    }        //k0
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////       END of GENERAL NULLING       ////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////      START of DigiCollections running:          ///////////////////////////////////
    ////////////////////////////////////////////////////////////////////
    if (flagupgradeqie1011_ != 2 && flagupgradeqie1011_ != 3 && flagupgradeqie1011_ != 6 && flagupgradeqie1011_ != 7 &&
        flagupgradeqie1011_ != 8) {
      edm::Handle<HFDigiCollection> hf;
      iEvent.getByToken(tok_hf_, hf);
      bool gotHFDigis = true;
      if (!(iEvent.getByToken(tok_hf_, hf))) {
        gotHFDigis = false;
      }  //this is a boolean set up to check if there are HFdigis in input root file
      if (!(hf.isValid())) {
        gotHFDigis = false;
      }  //if it is not there, leave it false
      if (!gotHFDigis) {
        cout << " ******************************  ===========================   No HFDigiCollection found " << endl;
      } else {
        ////////////////////////////////////////////////////////////////////   qie8   QIE8 :
        for (HFDigiCollection::const_iterator digi = hf->begin(); digi != hf->end(); digi++) {
          eta = digi->id().ieta();
          phi = digi->id().iphi();
          depth = digi->id().depth();
          nTS = digi->size();
          ///////////////////
          counterhf++;
          ////////////////////////////////////////////////////////////  for zerrors.C script:
          if (recordHistoes_ && studyCapIDErrorsHist_)
            fillDigiErrorsHF(digi);
          //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
          if (recordHistoes_)
            fillDigiAmplitudeHF(digi);
          //////////////////////////////////////////// calibration staff (often not needed):
          if (recordHistoes_ && studyCalibCellsHist_) {
            int iphi = phi - 1;
            int ieta = eta;
            if (ieta > 0)
              ieta -= 1;
            if (nTS <= numOfTS)
              for (int i = 0; i < nTS; i++) {
                TS_data[i] = adc2fC[digi->sample(i).adc()];
                signal[3][ieta + 41][iphi] += TS_data[i];
                if (i > 1 && i < 6)
                  signal3[3][ieta + 41][iphi] += TS_data[i];
              }  // TS
          }      // if(recordHistoes_ && studyCalibCellsHist_)
        }        // for
      }          // hf.isValid
    }            // end flagupgrade

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HFQIE10 DigiCollection
    //////////////////////////////////////////////////////////////////////////////////////////////////upgradeHF upgradehf
    // upgrade:
    if (flagupgradeqie1011_ != 1) {
      edm::Handle<QIE10DigiCollection> hfqie10;
      iEvent.getByToken(tok_qie10_, hfqie10);
      const QIE10DigiCollection& qie10dc =
          *(hfqie10);  ////////////////////////////////////////////////    <<<=========  !!!!
      bool gotQIE10Digis = true;
      if (!(iEvent.getByToken(tok_qie10_, hfqie10))) {
        gotQIE10Digis = false;
      }  //this is a boolean set up to check if there are HFdigis in input root file
      if (!(hfqie10.isValid())) {
        gotQIE10Digis = false;
      }  //if it is not there, leave it false
      if (!gotQIE10Digis) {
        cout << " No QIE10DigiCollection collection is found " << endl;
      } else {
        ////////////////////////////////////////////////////////////////////   qie10   QIE10 :
        double totalAmplitudeHF = 0.;
        for (unsigned int j = 0; j < qie10dc.size(); j++) {
          QIE10DataFrame qie10df = static_cast<QIE10DataFrame>(qie10dc[j]);
          DetId detid = qie10df.detid();
          HcalDetId hcaldetid = HcalDetId(detid);
          int eta = hcaldetid.ieta();
          int phi = hcaldetid.iphi();
          //	int depth = hcaldetid.depth();
          // loop over the samples in the digi
          nTS = qie10df.samples();
          ///////////////////
          counterhfqie10++;
          ////////////////////////////////////////////////////////////  for zerrors.C script:
          if (recordHistoes_ && studyCapIDErrorsHist_)
            fillDigiErrorsHFQIE10(qie10df);
          //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
          if (recordHistoes_)
            fillDigiAmplitudeHFQIE10(qie10df);
          ///////////////////
          //     if(recordHistoes_ ) {
          if (recordHistoes_ && studyCalibCellsHist_) {
            int iphi = phi - 1;
            int ieta = eta;
            if (ieta > 0)
              ieta -= 1;
            double amplitudefullTSs = 0.;
            double nnnnnnTS = 0.;
            for (int i = 0; i < nTS; ++i) {
              // j - QIE channel
              // i - time sample (TS)
              int adc = qie10df[i].adc();
              // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
              //	      float charge = adc2fC_QIE10[ adc ];
              TS_data[i] = adc2fC_QIE10[adc];
              signal[3][ieta + 41][iphi] += TS_data[i];
              totalAmplitudeHF += TS_data[i];
              amplitudefullTSs += TS_data[i];
              nnnnnnTS++;
              if (i > 1 && i < 6)
                signal3[3][ieta + 41][iphi] += TS_data[i];

            }  // TS
            h_numberofhitsHFtest->Fill(nnnnnnTS);
            h_AmplitudeHFtest->Fill(amplitudefullTSs);
          }  // if(recordHistoes_ && studyCalibCellsHist_)
        }    // for
        h_totalAmplitudeHF->Fill(totalAmplitudeHF);
        h_totalAmplitudeHFperEvent->Fill(float(eventcounter), totalAmplitudeHF);
      }  // hfqie10.isValid
    }    // end flagupgrade
    //end upgrade
    //
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// HBHEDigiCollection  usual, <=2018
    int qwert1 = 0;
    int qwert2 = 0;
    int qwert3 = 0;
    int qwert4 = 0;
    int qwert5 = 0;
    if (flagupgradeqie1011_ != 2 && flagupgradeqie1011_ != 3) {
      edm::Handle<HBHEDigiCollection> hbhe;
      iEvent.getByToken(tok_hbhe_, hbhe);
      bool gotHBHEDigis = true;
      if (!(iEvent.getByToken(tok_hbhe_, hbhe)))
        gotHBHEDigis = false;  //this is a boolean set up to check if there are HBHEgigis in input root file
      if (!(hbhe.isValid()))
        gotHBHEDigis = false;  //if it is not there, leave it false
      if (!gotHBHEDigis) {
        cout << " No HBHEDigiCollection collection is found " << endl;
      } else {
        //      unsigned int NHBHEDigiCollectionsize =  hbhe->size();
        double totalAmplitudeHB = 0.;
        double totalAmplitudeHE = 0.;
        double nnnnnnTSHB = 0.;
        double nnnnnnTSHE = 0.;

        for (HBHEDigiCollection::const_iterator digi = hbhe->begin(); digi != hbhe->end(); digi++) {
          eta = digi->id().ieta();
          phi = digi->id().iphi();
          depth = digi->id().depth();
          nTS = digi->size();
          /////////////////////////////////////// counters of event*digis
          nnnnnnhbhe++;
          nnnnnn++;
          //////////////////////////////////  counters of event for subdet & depth
          if (digi->id().subdet() == HcalBarrel && depth == 1 && qwert1 == 0) {
            nnnnnn1++;
            qwert1 = 1;
          }
          if (digi->id().subdet() == HcalBarrel && depth == 2 && qwert2 == 0) {
            nnnnnn2++;
            qwert2 = 1;
          }
          if (digi->id().subdet() == HcalEndcap && depth == 1 && qwert3 == 0) {
            nnnnnn3++;
            qwert3 = 1;
          }
          if (digi->id().subdet() == HcalEndcap && depth == 2 && qwert4 == 0) {
            nnnnnn4++;
            qwert4 = 1;
          }
          if (digi->id().subdet() == HcalEndcap && depth == 3 && qwert5 == 0) {
            nnnnnn5++;
            qwert5 = 1;
          }
          ////////////////////////////////////////////////////////////  for zerrors.C script:
          if (recordHistoes_ && studyCapIDErrorsHist_)
            fillDigiErrors(digi);
          //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
          if (recordHistoes_)
            fillDigiAmplitude(digi);

          if (recordHistoes_ && studyCalibCellsHist_) {
            int iphi = phi - 1;
            int ieta = eta;
            if (ieta > 0)
              ieta -= 1;
            //////////////////////////////////////////    HB:
            if (digi->id().subdet() == HcalBarrel) {
              double amplitudefullTSs = 0.;
              nnnnnnTSHB++;
              if (nTS <= numOfTS)
                for (int i = 0; i < nTS; i++) {
                  TS_data[i] = adc2fC[digi->sample(i).adc()];
                  signal[0][ieta + 41][iphi] += TS_data[i];
                  amplitudefullTSs += TS_data[i];
                  totalAmplitudeHB += TS_data[i];
                  if (i > 1 && i < 6)
                    signal3[0][ieta + 41][iphi] += TS_data[i];
                }
              h_AmplitudeHBtest->Fill(amplitudefullTSs);
            }  // HB
            //////////////////////////////////////////    HE:
            if (digi->id().subdet() == HcalEndcap) {
              double amplitudefullTSs = 0.;
              nnnnnnTSHE++;
              if (nTS <= numOfTS)
                for (int i = 0; i < nTS; i++) {
                  TS_data[i] = adc2fC[digi->sample(i).adc()];
                  signal[1][ieta + 41][iphi] += TS_data[i];
                  totalAmplitudeHE += TS_data[i];
                  amplitudefullTSs += TS_data[i];
                  if (i > 1 && i < 6)
                    signal3[1][ieta + 41][iphi] += TS_data[i];
                }
              h_AmplitudeHEtest->Fill(amplitudefullTSs);
            }  // HE

          }  //if(recordHistoes_ && studyCalibCellsHist_)
          if (recordNtuples_ && nevent50 < maxNeventsInNtuple_) {
          }  //if(recordNtuples_)
        }    // for HBHE digis
        if (totalAmplitudeHB != 0.) {
          h_numberofhitsHBtest->Fill(nnnnnnTSHB);
          h_totalAmplitudeHB->Fill(totalAmplitudeHB);
          h_totalAmplitudeHBperEvent->Fill(float(eventcounter), totalAmplitudeHB);
        }
        if (totalAmplitudeHE != 0.) {
          h_numberofhitsHEtest->Fill(nnnnnnTSHE);
          h_totalAmplitudeHE->Fill(totalAmplitudeHE);
          h_totalAmplitudeHEperEvent->Fill(float(eventcounter), totalAmplitudeHE);
        }
      }  //hbhe.isValid
    }    // end flagupgrade
    //---------------------------------------------------------------
    //////////////////////////////////////////////////////////////////////////////////////////////////    upgradeHBHE upgradehe       HBHE with SiPM (both >=2020)
    // upgrade:
    if (flagupgradeqie1011_ != 1 && flagupgradeqie1011_ != 4 && flagupgradeqie1011_ != 5 && flagupgradeqie1011_ != 10) {
      edm::Handle<QIE11DigiCollection> heqie11;
      iEvent.getByToken(tok_qie11_, heqie11);
      const QIE11DigiCollection& qie11dc =
          *(heqie11);  ////////////////////////////////////////////////    <<<=========  !!!!
      bool gotQIE11Digis = true;
      if (!(iEvent.getByToken(tok_qie11_, heqie11)))
        gotQIE11Digis = false;  //this is a boolean set up to check if there are QIE11gigis in input root file
      if (!(heqie11.isValid()))
        gotQIE11Digis = false;  //if it is not there, leave it false
      if (!gotQIE11Digis) {
        cout << " No QIE11DigiCollection collection is found " << endl;
      } else {
        ////////////////////////////////////////////////////////////////////   qie11   QIE11 :
        double totalAmplitudeHBQIE11 = 0.;
        double totalAmplitudeHEQIE11 = 0.;
        double nnnnnnTSHBQIE11 = 0.;
        double nnnnnnTSHEQIE11 = 0.;
        for (unsigned int j = 0; j < qie11dc.size(); j++) {
          QIE11DataFrame qie11df = static_cast<QIE11DataFrame>(qie11dc[j]);
          DetId detid = qie11df.detid();
          HcalDetId hcaldetid = HcalDetId(detid);
          int eta = hcaldetid.ieta();
          int phi = hcaldetid.iphi();
          int depth = hcaldetid.depth();
          if (depth == 0)
            return;
          int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
          // loop over the samples in the digi
          nTS = qie11df.samples();
          ///////////////////
          nnnnnnhbheqie11++;
          nnnnnn++;
          if (recordHistoes_ && studyCapIDErrorsHist_)
            fillDigiErrorsQIE11(qie11df);
          //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
          if (recordHistoes_)
            fillDigiAmplitudeQIE11(qie11df);
          ///////////////////
          //////////////////////////////////  counters of event for subdet & depth
          if (sub == 1 && depth == 1 && qwert1 == 0) {
            nnnnnn1++;
            qwert1 = 1;
          }
          if (sub == 1 && depth == 2 && qwert2 == 0) {
            nnnnnn2++;
            qwert2 = 1;
          }
          if (sub == 2 && depth == 1 && qwert3 == 0) {
            nnnnnn3++;
            qwert3 = 1;
          }
          if (sub == 2 && depth == 2 && qwert4 == 0) {
            nnnnnn4++;
            qwert4 = 1;
          }
          if (sub == 2 && depth == 3 && qwert5 == 0) {
            nnnnnn5++;
            qwert5 = 1;
          }

          if (recordHistoes_ && studyCalibCellsHist_) {
            int iphi = phi - 1;
            int ieta = eta;
            if (ieta > 0)
              ieta -= 1;
            // HB:
            if (sub == 1) {
              double amplitudefullTSs1 = 0.;
              double amplitudefullTSs6 = 0.;
              nnnnnnTSHBQIE11++;
              for (int i = 0; i < nTS; ++i) {
                int adc = qie11df[i].adc();
                double charge1 = adc2fC_QIE11_shunt1[adc];
                double charge6 = adc2fC_QIE11_shunt6[adc];
                amplitudefullTSs1 += charge1;
                amplitudefullTSs6 += charge6;
                double charge = charge6;
                TS_data[i] = charge;
                signal[0][ieta + 41][iphi] += charge;
                if (i > 1 && i < 6)
                  signal3[0][ieta + 41][iphi] += charge;
                totalAmplitudeHBQIE11 += charge;
              }  //for
              h_AmplitudeHBtest1->Fill(amplitudefullTSs1, 1.);
              h_AmplitudeHBtest6->Fill(amplitudefullTSs6, 1.);
            }  //HB end
            // HE:
            if (sub == 2) {
              double amplitudefullTSs1 = 0.;
              double amplitudefullTSs6 = 0.;
              nnnnnnTSHEQIE11++;
              for (int i = 0; i < nTS; i++) {
                int adc = qie11df[i].adc();
                double charge1 = adc2fC_QIE11_shunt1[adc];
                double charge6 = adc2fC_QIE11_shunt6[adc];
                amplitudefullTSs1 += charge1;
                amplitudefullTSs6 += charge6;
                double charge = charge6;
                TS_data[i] = charge;
                signal[1][ieta + 41][iphi] += charge;
                if (i > 1 && i < 6)
                  signal3[1][ieta + 41][iphi] += charge;
                totalAmplitudeHEQIE11 += charge;
              }  //for
              h_AmplitudeHEtest1->Fill(amplitudefullTSs1, 1.);
              h_AmplitudeHEtest6->Fill(amplitudefullTSs6, 1.);

            }  //HE end
          }    //if(recordHistoes_ && studyCalibCellsHist_)
        }      // for QIE11 digis

        if (totalAmplitudeHBQIE11 != 0.) {
          h_numberofhitsHBtest->Fill(nnnnnnTSHBQIE11);
          h_totalAmplitudeHB->Fill(totalAmplitudeHBQIE11);
          h_totalAmplitudeHBperEvent->Fill(float(eventcounter), totalAmplitudeHBQIE11);
        }
        if (totalAmplitudeHEQIE11 != 0.) {
          h_numberofhitsHEtest->Fill(nnnnnnTSHEQIE11);
          h_totalAmplitudeHE->Fill(totalAmplitudeHEQIE11);
          h_totalAmplitudeHEperEvent->Fill(float(eventcounter), totalAmplitudeHEQIE11);
        }
      }  //heqie11.isValid
    }    // end flagupgrade

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////   HODigiCollection
    edm::Handle<HODigiCollection> ho;
    iEvent.getByToken(tok_ho_, ho);
    bool gotHODigis = true;
    if (!(iEvent.getByToken(tok_ho_, ho)))
      gotHODigis = false;  //this is a boolean set up to check if there are HOgigis in input root file
    if (!(ho.isValid()))
      gotHODigis = false;  //if it is not there, leave it false
    if (!gotHODigis) {
      //  if(!ho.isValid()) {
      cout << " No HO collection is found " << endl;
    } else {
      int qwert6 = 0;
      double totalAmplitudeHO = 0.;
      for (HODigiCollection::const_iterator digi = ho->begin(); digi != ho->end(); digi++) {
        eta = digi->id().ieta();
        phi = digi->id().iphi();
        depth = digi->id().depth();
        nTS = digi->size();
        ///////////////////
        counterho++;
        //////////////////////////////////  counters of event
        if (qwert6 == 0) {
          nnnnnn6++;
          qwert6 = 1;
        }
        ////////////////////////////////////////////////////////////  for zerrors.C script:
        if (recordHistoes_ && studyCapIDErrorsHist_)
          fillDigiErrorsHO(digi);
        //////////////////////////////////////  for ztsmaxa.C,zratio34.C,zrms.C & zdifampl.C scripts:
        if (recordHistoes_)
          fillDigiAmplitudeHO(digi);
        ///////////////////
        if (recordHistoes_ && studyCalibCellsHist_) {
          int iphi = phi - 1;
          int ieta = eta;
          if (ieta > 0)
            ieta -= 1;
          double nnnnnnTS = 0.;
          double amplitudefullTSs = 0.;
          if (nTS <= numOfTS)
            for (int i = 0; i < nTS; i++) {
              TS_data[i] = adc2fC[digi->sample(i).adc()];
              amplitudefullTSs += TS_data[i];
              signal[2][ieta + 41][iphi] += TS_data[i];
              totalAmplitudeHO += TS_data[i];
              if (i > 1 && i < 6)
                signal3[2][ieta + 41][iphi] += TS_data[i];
              nnnnnnTS++;
            }  //if for
          h_AmplitudeHOtest->Fill(amplitudefullTSs);
          h_numberofhitsHOtest->Fill(nnnnnnTS);
        }  //if(recordHistoes_ && studyCalibCellsHist_)
      }    //for HODigiCollection

      h_totalAmplitudeHO->Fill(totalAmplitudeHO);
      h_totalAmplitudeHOperEvent->Fill(float(eventcounter), totalAmplitudeHO);
    }  //ho.isValid(

    //////////////////////////////////////////////////////
    if (flagLaserRaddam_ > 1) {
      ////////////////////////////////////////////////////// RADDAM treatment:
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            if (mapRADDAM0_HE[k1][k2][k3] != 0) {
              // ----------------------------------------    D2 sum over phi before!!! any dividing:
              mapRADDAM_HED2[k1][k2] += mapRADDAM_HE[k1][k2][k3];
              // N phi sectors w/ digihits
              ++mapRADDAM_HED20[k1][k2];
            }  //if
          }    //for
        }      //for
      }        //for

      //////////////---------------------------------------------------------------------------------  2D treatment, zraddam2.cc script
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (mapRADDAM_HED20[k1][k2] != 0) {
            // validation of channels at eta16:
            if (k1 == 2 && k2 == 25) {
              h_sumphiEta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2->Fill(mapRADDAM_HED20[k1][k2]);
            } else if (k1 == 2 && k2 == 56) {
              h_sumphiEta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2P->Fill(mapRADDAM_HED20[k1][k2]);
            } else {
              h_sumphiEta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED2[k1][k2]);
              h_Eta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED2[k1][k2] / mapRADDAM_HED20[k1][k2]);
              h_NphiForEta16Depth3RADDAM_HED2ALL->Fill(mapRADDAM_HED20[k1][k2]);
            }
            //////////////-----------------------  aver per N-phi_sectors ???
            mapRADDAM_HED2[k1][k2] /= mapRADDAM_HED20[k1][k2];
          }  // if(mapRADDAM_HED20[k1][k2] != 0
        }    //for
      }      //for
      ///////////////////////////////////////////
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (k1 == 2 && (k2 == 25 || k2 == 56)) {
          } else {
            //	if(k2!=25 && k2!=56) {
            int k2plot = k2 - 41;
            int kkk = k2;
            if (k2plot > 0)
              kkk = k2 + 1;
            int kk2 = 25;
            if (k2plot > 0)
              kk2 = 56;
            if (mapRADDAM_HED2[k1][k2] != 0. && mapRADDAM_HED2[2][kk2] != 0) {
              mapRADDAM_HED2[k1][k2] /= mapRADDAM_HED2[2][kk2];
              // (d1 & eta 17-29)                       L1
              int LLLLLL111111 = 0;
              if ((k1 == 0 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 30))
                LLLLLL111111 = 1;
              // (d2 & eta 17-26) && (d3 & eta 27-28)   L2
              int LLLLLL222222 = 0;
              if ((k1 == 1 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 27) ||
                  (k1 == 2 && fabs(kkk - 41) > 26 && fabs(kkk - 41) < 29))
                LLLLLL222222 = 1;
              //
              if (LLLLLL111111 == 1) {
                h_sigLayer1RADDAM5_HED2->Fill(double(kkk - 41), mapRADDAM_HED2[k1][k2]);
                h_sigLayer1RADDAM6_HED2->Fill(double(kkk - 41), 1.);
              }
              if (LLLLLL222222 == 1) {
                h_sigLayer2RADDAM5_HED2->Fill(double(kkk - 41), mapRADDAM_HED2[k1][k2]);
                h_sigLayer2RADDAM6_HED2->Fill(double(kkk - 41), 1.);
              }
            }  //if
          }    // if(k2!=25 && k2!=56
        }      //for
      }        //for

      //////////////---------------------------------------------------------------------------------  3D treatment, zraddam1.cc script

      //------------------------------------------------------        aver per eta 16(depth=3-> k1=2, k2=16(15) :
      //////////// k1(depth-1): = 0 - 6 or depth: = 1 - 7;
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          if (k1 == 2 && (k2 == 25 || k2 == 56)) {
          } else {
            int k2plot = k2 - 41;
            int kk2 = 25;
            if (k2plot > 0)
              kk2 = 56;
            int kkk = k2;
            if (k2plot > 0)
              kkk = k2 + 1;
            for (int k3 = 0; k3 < nphi; k3++) {
              if (mapRADDAM_HE[k1][k2][k3] != 0. && mapRADDAM_HE[2][kk2][k3] != 0) {
                mapRADDAM_HE[k1][k2][k3] /= mapRADDAM_HE[2][kk2][k3];
                int LLLLLL111111 = 0;
                if ((k1 == 0 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 30))
                  LLLLLL111111 = 1;
                int LLLLLL222222 = 0;
                if ((k1 == 1 && fabs(kkk - 41) > 16 && fabs(kkk - 41) < 27) ||
                    (k1 == 2 && fabs(kkk - 41) > 26 && fabs(kkk - 41) < 29))
                  LLLLLL222222 = 1;
                if (LLLLLL111111 == 1) {
                  h_sigLayer1RADDAM5_HE->Fill(double(kkk - 41), mapRADDAM_HE[k1][k2][k3]);
                  h_sigLayer1RADDAM6_HE->Fill(double(kkk - 41), 1.);
                }
                if (LLLLLL222222 == 1) {
                  h_sigLayer2RADDAM5_HE->Fill(double(kkk - 41), mapRADDAM_HE[k1][k2][k3]);
                  h_sigLayer2RADDAM6_HE->Fill(double(kkk - 41), 1.);
                }
              }  //if
            }    //for
          }      // if(k2!=25 && k2!=56
        }        //for
      }          //for
                 //
                 ////////////////////////////////////////////////////////////////////////////////////////////////
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          mapRADDAM_HED2[k1][k2] = 0.;
          mapRADDAM_HED20[k1][k2] = 0.;
          for (int k3 = 0; k3 < nphi; k3++) {
            mapRADDAM_HE[k1][k2][k3] = 0.;
            mapRADDAM0_HE[k1][k2][k3] = 0;
          }  //for
        }    //for
      }      //for

      //////////////////////////////////END of RADDAM treatment:
    }  // END TREATMENT : if(flagLaserRaddam_ == 1
    //////////////////////////////////////////////////////////////////////////
    //////////// k0(sub):       =0 HB;      =1 HE;       =2 HO;       =3 HF;
    //////////// k1(depth-1): = 0 - 6 or depth: = 1 - 7;

    if (flagIterativeMethodCalibrationGroup_ > 0) {
      //////////////////////////////////////////////////////////////////////////
      //	  //	  //	  // tocdefault tocampl tocamplchannel: calibration group, Iterative method, coding start 29.08.2019
      //
      for (int k0 = 0; k0 < nsub; k0++) {
        // HE only, temporary
        if (k0 == 1) {
          for (int k1 = 0; k1 < ndepth; k1++) {
            // k2: 0-81
            for (int k2 = 0; k2 < neta; k2++) {
              int k2plot = k2 - 41;
              int kkk = k2plot;
              if (k2plot > 0)
                kkk = k2plot + 1;  //-41 +41 !=0

              //preparation for PHI normalization:
              double sumoverphi = 0;
              int nsumoverphi = 0;
              for (int k3 = 0; k3 < nphi; k3++) {
                if (tocamplchannel[k0][k1][k2][k3] != 0) {
                  sumoverphi += tocamplchannel[k0][k1][k2][k3];
                  ++nsumoverphi;
                }  //if != 0
              }    //k3

              // PHI normalization into new massive && filling plots:
              for (int k3 = 0; k3 < nphi; k3++) {
                if (nsumoverphi != 0) {
                  maprphinorm[k0][k1][k2][k3] = tocamplchannel[k0][k1][k2][k3] / (sumoverphi / nsumoverphi);
                  // filling plots:
                  if (k0 == 1 && k1 == 0) {
                    h_mapenophinorm_HE1->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE1->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE1->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE1->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE1->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 1) {
                    h_mapenophinorm_HE2->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE2->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE2->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE2->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE2->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 2) {
                    h_mapenophinorm_HE3->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE3->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE3->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE3->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE3->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 3) {
                    h_mapenophinorm_HE4->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE4->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE4->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE4->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE4->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 4) {
                    h_mapenophinorm_HE5->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE5->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE5->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE5->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE5->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 5) {
                    h_mapenophinorm_HE6->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE6->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE6->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE6->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE6->Fill(double(kkk), double(k3), 1.);
                  }
                  if (k0 == 1 && k1 == 6) {
                    h_mapenophinorm_HE7->Fill(double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3]);
                    h_mapenophinorm2_HE7->Fill(
                        double(kkk), double(k3), tocamplchannel[k0][k1][k2][k3] * tocamplchannel[k0][k1][k2][k3]);
                    h_maprphinorm_HE7->Fill(double(kkk), double(k3), maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm2_HE7->Fill(
                        double(kkk), double(k3), maprphinorm[k0][k1][k2][k3] * maprphinorm[k0][k1][k2][k3]);
                    h_maprphinorm0_HE7->Fill(double(kkk), double(k3), 1.);
                  }
                }  //if nsumoverphi != 0
              }    //k3
            }      //k2
          }        //k1
        }          //if k0 == 1 HE only, temporary
      }            //k0

      // tocampl end
    }  // if(flagIterativeMethodCalibrationGroup

    //////////////////////////////////////////////////////////////////////////
    //	  //	  //	  //	  //	  //	  //sumamplitudes:
    int testcount1 = 0;
    int testcount2 = 0;
    int testcount3 = 0;
    ////////////////////////////////////////  // k0, k2, k3 loops LOOPS  //////////   /////  ///// NO k1 loop over depthes !!!
    for (int k0 = 0; k0 < nsub; k0++) {
      int sumofchannels = 0;
      double sumamplitudesubdet = 0.;
      int sumofchannels0 = 0;
      double sumamplitudesubdet0 = 0.;
      for (int k2 = 0; k2 < neta; k2++) {
        for (int k3 = 0; k3 < nphi; k3++) {
          if (amplitudechannel[k0][0][k2][k3] != 0. && amplitudechannel[k0][1][k2][k3] != 0.)
            testcount1++;
          if (amplitudechannel[k0][0][k2][k3] != 0. && amplitudechannel[k0][1][k2][k3] == 0.)
            testcount2++;
          if (amplitudechannel[k0][0][k2][k3] == 0. && amplitudechannel[k0][1][k2][k3] != 0.)
            testcount3++;

          // HB
          if (k0 == 0) {
            double sumamplitudechannel_HB = amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3];
            h_sumamplitudechannel_HB->Fill(sumamplitudechannel_HB);
            if (sumamplitudechannel_HB > 80.) {
              sumamplitudesubdet += sumamplitudechannel_HB;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HB > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HB;
                sumofchannels0++;
              }
            }
          }  //

          // HE
          if (k0 == 1) {
            double sumamplitudechannel_HE =
                amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3] + amplitudechannel[k0][2][k2][k3];
            h_sumamplitudechannel_HE->Fill(sumamplitudechannel_HE);
            if (sumamplitudechannel_HE > 200.) {
              sumamplitudesubdet += sumamplitudechannel_HE;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HE > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HE;
                sumofchannels0++;
              }
            }
          }  //

          // HO
          if (k0 == 2) {
            double sumamplitudechannel_HO = amplitudechannel[k0][3][k2][k3];
            h_sumamplitudechannel_HO->Fill(sumamplitudechannel_HO);
            if (sumamplitudechannel_HO > 1200.) {
              sumamplitudesubdet += sumamplitudechannel_HO;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HO > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HO;
                sumofchannels0++;
              }
            }
          }  //
          // HF
          if (k0 == 3) {
            double sumamplitudechannel_HF = amplitudechannel[k0][0][k2][k3] + amplitudechannel[k0][1][k2][k3];
            h_sumamplitudechannel_HF->Fill(sumamplitudechannel_HF);
            if (sumamplitudechannel_HF > 600.) {
              sumamplitudesubdet += sumamplitudechannel_HF;
              sumofchannels++;
            } else {
              if (sumamplitudechannel_HF > 0.) {
                sumamplitudesubdet0 += sumamplitudechannel_HF;
                sumofchannels0++;
              }
            }
          }  //

        }  //k3
      }    //k2
      //  }//k1
      // SA of each sub-detector DONE. Then: summarize or find maximum throught events of LS
      if (k0 == 0) {
        h_eventamplitude_HB->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HB->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM1)
          maxxSUM1 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP1)
          maxxOCCUP1 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HB += sumofchannels;
        averSIGNALsumamplitude_HB += sumamplitudesubdet;
        averNOSIGNALoccupancy_HB += sumofchannels0;
        averNOSIGNALsumamplitude_HB += sumamplitudesubdet0;
        if ((sumamplitudesubdet + sumamplitudesubdet0) > 60000) {
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              int ieta = k2 - 41;
              /// HB depth1:
              if (amplitudechannel[k0][0][k2][k3] != 0.) {
                h_2DAtaildepth1_HB->Fill(double(ieta), double(k3), amplitudechannel[k0][0][k2][k3]);
                h_2D0Ataildepth1_HB->Fill(double(ieta), double(k3), 1.);
              }
              /// HB depth2:
              if (amplitudechannel[k0][1][k2][k3] != 0.) {
                h_2DAtaildepth2_HB->Fill(double(ieta), double(k3), amplitudechannel[k0][1][k2][k3]);
                h_2D0Ataildepth2_HB->Fill(double(ieta), double(k3), 1.);
              }
            }  //for
          }    //for
        }      //>60000
      }        //HB
      if (k0 == 1) {
        h_eventamplitude_HE->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HE->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM2)
          maxxSUM2 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP2)
          maxxOCCUP2 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HE += sumofchannels;
        averSIGNALsumamplitude_HE += sumamplitudesubdet;
        averNOSIGNALoccupancy_HE += sumofchannels0;
        averNOSIGNALsumamplitude_HE += sumamplitudesubdet0;
      }  //HE
      if (k0 == 2) {
        h_eventamplitude_HO->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HO->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM3)
          maxxSUM3 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP3)
          maxxOCCUP3 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HO += sumofchannels;
        averSIGNALsumamplitude_HO += sumamplitudesubdet;
        averNOSIGNALoccupancy_HO += sumofchannels0;
        averNOSIGNALsumamplitude_HO += sumamplitudesubdet0;
      }  //HO
      if (k0 == 3) {
        h_eventamplitude_HF->Fill((sumamplitudesubdet + sumamplitudesubdet0));
        h_eventoccupancy_HF->Fill((sumofchannels + sumofchannels0));
        if ((sumamplitudesubdet + sumamplitudesubdet0) > maxxSUM4)
          maxxSUM4 = sumamplitudesubdet + sumamplitudesubdet0;
        if ((sumofchannels + sumofchannels0) > maxxOCCUP4)
          maxxOCCUP4 = sumofchannels + sumofchannels0;
        averSIGNALoccupancy_HF += sumofchannels;
        averSIGNALsumamplitude_HF += sumamplitudesubdet;
        averNOSIGNALoccupancy_HF += sumofchannels0;
        averNOSIGNALsumamplitude_HF += sumamplitudesubdet0;
      }  //HF
    }    //k0

    ///////////////////// ///////////////////// //////////////////////////////////////////
    ///////////////////////////////////////////////  for zRunRatio34.C & zRunNbadchan.C scripts:
    if (recordHistoes_ && studyRunDependenceHist_) {
      int eeeeee;
      eeeeee = lscounterM1;
      if (flagtoaskrunsorls_ == 0)
        eeeeee = runcounter;

      //////////// k0(sub): =0 HB; =1 HE; =2 HO; =3 HF;
      for (int k0 = 0; k0 < nsub; k0++) {
        //////////// k1(depth-1): = 0 - 3 or depth: = 1 - 4;
        //////////// for upgrade    k1(depth-1): = 0 - 6 or depth: = 1 - 7;
        for (int k1 = 0; k1 < ndepth; k1++) {
          //////////
          int nbadchannels = 0;
          for (int k2 = 0; k2 < neta; k2++) {
            for (int k3 = 0; k3 < nphi; k3++) {
              if (badchannels[k0][k1][k2][k3] != 0)
                ++nbadchannels;
            }  //k3
          }    //k2
          //////////
          //HB
          if (k0 == 0) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HB->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HBdepth1_)
                h_runnbadchannelsC_depth1_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HB->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HBdepth1_)
                h_runbadrateC_depth1_HB->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HB->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HB->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HB->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HB->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HBdepth2_)
                h_runnbadchannelsC_depth2_HB->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HB->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HBdepth2_)
                h_runbadrateC_depth2_HB->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HB->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HB->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HB->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 0)
          //HE
          if (k0 == 1) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth1_)
                h_runnbadchannelsC_depth1_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth1_)
                h_runbadrateC_depth1_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HE->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth2_)
                h_runnbadchannelsC_depth2_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth2_)
                h_runbadrateC_depth2_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HE->Fill(float(bcn), 1.);
            }
            if (k1 == 2) {
              h_nbadchannels_depth3_HE->Fill(float(nbadchannels));
              h_runnbadchannels_depth3_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HEdepth3_)
                h_runnbadchannelsC_depth3_HE->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth3_HE->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HEdepth3_)
                h_runbadrateC_depth3_HE->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth3_HE->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth3_HE->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth3_HE->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 1)
          //HO
          if (k0 == 2) {
            if (k1 == 3) {
              h_nbadchannels_depth4_HO->Fill(float(nbadchannels));
              h_runnbadchannels_depth4_HO->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HOdepth4_)
                h_runnbadchannelsC_depth4_HO->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth4_HO->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HOdepth4_)
                h_runbadrateC_depth4_HO->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth4_HO->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth4_HO->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth4_HO->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 2)
          //HF
          if (k0 == 3) {
            if (k1 == 0) {
              h_nbadchannels_depth1_HF->Fill(float(nbadchannels));
              h_runnbadchannels_depth1_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HFdepth1_)
                h_runnbadchannelsC_depth1_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth1_HF->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HFdepth1_)
                h_runbadrateC_depth1_HF->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth1_HF->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth1_HF->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth1_HF->Fill(float(bcn), 1.);
            }
            if (k1 == 1) {
              h_nbadchannels_depth2_HF->Fill(float(nbadchannels));
              h_runnbadchannels_depth2_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels > lsdep_cut1_peak_HFdepth2_)
                h_runnbadchannelsC_depth2_HF->Fill(float(eeeeee), float(nbadchannels));
              if (nbadchannels != 0)
                h_runbadrate_depth2_HF->Fill(float(eeeeee), 1.);
              if (nbadchannels > lsdep_cut3_max_HFdepth2_)
                h_runbadrateC_depth2_HF->Fill(float(eeeeee), 1.);
              h_bcnnbadchannels_depth2_HF->Fill(float(bcn), float(nbadchannels));
              h_runbadrate0_depth2_HF->Fill(float(eeeeee), 1.);
              h_bcnbadrate0_depth2_HF->Fill(float(bcn), 1.);
            }
          }  ////if(k0 == 3)

          //////////
        }  //k1
      }    //k0
      ////////////
    }  //if(recordHistoes_&& studyRunDependenceHist_)

    /////////////////////////////////////////////////////////////////////////////////////// HcalCalibDigiCollection
    edm::Handle<HcalCalibDigiCollection> calib;
    iEvent.getByToken(tok_calib_, calib);

    bool gotCALIBDigis = true;
    if (!(iEvent.getByToken(tok_calib_, calib))) {
      gotCALIBDigis = false;  //this is a boolean set up to check if there are CALIBgigis in input root file
    }
    if (!(calib.isValid())) {
      gotCALIBDigis = false;  //if it is not there, leave it false
    }
    if (!gotCALIBDigis) {
    } else {
      for (HcalCalibDigiCollection::const_iterator digi = calib->begin(); digi != calib->end(); digi++) {
        int cal_det = digi->id().hcalSubdet();  // 1-HB,2-HE,3-HO,4-HF
        int cal_phi = digi->id().iphi();
        int cal_eta = digi->id().ieta();
        int cal_cbox = digi->id().cboxChannel();

        /////////////////////////////////////////////
        if (recordHistoes_ && studyCalibCellsHist_) {
          if (cal_det > 0 && cal_det < 5 && cal_cbox == 0) {
            int iphi = cal_phi - 1;
            int ieta = cal_eta;
            if (ieta > 0)
              ieta -= 1;
            nTS = digi->size();
            double max_signal = -100.;
            int ts_with_max_signal = -100;
            double timew = 0.;

            //
            if (nTS <= numOfTS)
              for (int i = 0; i < nTS; i++) {
                double ampldefault = adc2fC[digi->sample(i).adc() & 0xff];
                if (max_signal < ampldefault) {
                  max_signal = ampldefault;
                  ts_with_max_signal = i;
                }
                if (i > 1 && i < 6)
                  calib3[cal_det - 1][ieta + 41][iphi] += ampldefault;
                calib0[cal_det - 1][ieta + 41][iphi] += ampldefault;
                timew += (i + 1) * ampldefault;
              }  // for
            //

            double amplitude = calib0[cal_det - 1][ieta + 41][iphi];
            double aveamplitude = -100.;
            if (amplitude > 0 && timew > 0)
              aveamplitude = timew / amplitude;       // average_TS +1
            double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9
            caliba[cal_det - 1][ieta + 41][iphi] = aveamplitude1;

            double rmsamp = 0.;
            for (int ii = 0; ii < nTS; ii++) {
              double ampldefault = adc2fC[digi->sample(ii).adc() & 0xff];
              double aaaaaa = (ii + 1) - aveamplitude;
              double aaaaaa2 = aaaaaa * aaaaaa;
              rmsamp += (aaaaaa2 * ampldefault);  // fC
            }                                     //for 2
            double rmsamplitude = -100.;
            if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
              rmsamplitude = sqrt(rmsamp / amplitude);
            calibw[cal_det - 1][ieta + 41][iphi] = rmsamplitude;
            //
            calibt[cal_det - 1][ieta + 41][iphi] = ts_with_max_signal;
            //

            if (ts_with_max_signal > -1 && ts_with_max_signal < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] = adc2fC[digi->sample(ts_with_max_signal).adc() & 0xff];
            if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal + 1).adc() & 0xff];
            if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal + 2).adc() & 0xff];
            if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal - 1).adc() & 0xff];
            if (ts_with_max_signal - 2 > -1 && ts_with_max_signal - 2 < nTS)
              calib2[cal_det - 1][ieta + 41][iphi] += adc2fC[digi->sample(ts_with_max_signal - 2).adc() & 0xff];
            //
            bool anycapid = true;
            bool anyer = false;
            bool anydv = true;
            int error1 = 0, error2 = 0, error3 = 0;
            int lastcapid = 0, capid = 0;
            for (int ii = 0; ii < (*digi).size(); ii++) {
              capid = (*digi)[ii].capid();  // capId (0-3, sequential)
              bool er = (*digi)[ii].er();   // error
              bool dv = (*digi)[ii].dv();   // valid data
              if (ii != 0 && ((lastcapid + 1) % 4) != capid)
                anycapid = false;
              lastcapid = capid;
              if (er)
                anyer = true;
              if (!dv)
                anydv = false;
            }
            if (!anycapid)
              error1 = 1;
            if (anyer)
              error2 = 1;
            if (!anydv)
              error3 = 1;
            if (error1 != 0 || error2 != 0 || error3 != 0)
              calibcapiderror[cal_det - 1][ieta + 41][iphi] = 100;

          }  // if(cal_det>0 && cal_det<5
        }    //if(recordHistoes_ && studyCalibCellsHist_)
        /////////////////////////////////////////////

        if (recordNtuples_ && nevent50 < maxNeventsInNtuple_) {
        }  //if(recordNtuples_) {

      }  //for(HcalCalibDigiCollection
    }    //if(calib.isValid(
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (recordHistoes_ && studyCalibCellsHist_) {
      ////////////////////////////////////for loop for zcalib.C and zgain.C scripts:
      for (int k1 = 0; k1 < nsub; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            int k2plot = k2 - 41;
            if (flagcpuoptimization_ == 0) {
              ////////////////////////////////////////////////////////////////  for zgain.C script:

              if (signal[k1][k2][k3] > 0.) {
                if (k1 == 0) {
                  h_FullSignal3D_HB->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HB->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 1) {
                  h_FullSignal3D_HE->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HE->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 2) {
                  h_FullSignal3D_HO->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HO->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 3) {
                  h_FullSignal3D_HF->Fill(double(k2plot), double(k3), signal[k1][k2][k3]);
                  h_FullSignal3D0_HF->Fill(double(k2plot), double(k3), 1.);
                }
              }

            }  // optimization
            ////////////////////////////////////////////////////////////////

            ////////////////////////////////////////////////////////////////  for zcalib.C script:
            // k2 = 0-81, k3= 0-71
            // return to real indexes in eta and phi ( k20 and k30)
            int k20 = k2 - 41;  // k20 = -41 - 40
            if (k20 > 0 || k20 == 0)
              k20 += 1;        // k20 = -41 - -1 and +1 - +41
            int k30 = k3 + 1;  // k30= 1-nphi

            // find calibration indexes in eta and phi ( kk2 and kk3)
            int kk2 = 0, kk3 = 0;
            if (k1 == 0 || k1 == 1) {
              if (k20 > 0)
                kk2 = 1;
              else
                kk2 = -1;
              if (k30 == 71 || k30 == nphi || k30 == 1 || k30 == 2)
                kk3 = 71;
              else
                kk3 = ((k30 - 3) / 4) * 4 + 3;
            } else if (k1 == 2) {
              if (abs(k20) <= 4) {
                kk2 = 0;
                if (k30 == 71 || k30 == nphi || k30 == 1 || k30 == 2 || k30 == 3 || k30 == 4)
                  kk3 = 71;
                else
                  kk3 = ((k30 - 5) / 6) * 6 + 5;
              } else {
                if (abs(k20) > 4 && abs(k20) <= 10)
                  kk2 = 1;
                if (abs(k20) > 10 && abs(k20) <= 15)
                  kk2 = 2;
                if (k20 < 0)
                  kk2 = -kk2;
                if (k30 == 71 || k30 == nphi || (k30 >= 1 && k30 <= 10))
                  kk3 = 71;
                else
                  kk3 = ((k30 - 11) / 12) * 12 + 11;
              }
            } else if (k1 == 3) {
              if (k20 > 0)
                kk2 = 1;
              else
                kk2 = -1;
              if (k30 >= 1 && k30 <= 18)
                kk3 = 1;
              if (k30 >= 19 && k30 <= 36)
                kk3 = 19;
              if (k30 >= 37 && k30 <= 54)
                kk3 = 37;
              if (k30 >= 55 && k30 <= nphi)
                kk3 = 55;
            }
            // return to indexes in massiv
            int kkk2 = kk2 + 41;
            if (kk2 > 0)
              kkk2 -= 1;
            int kkk3 = kk3;
            kkk3 -= 1;

            if (flagcpuoptimization_ == 0) {
              double GetRMSOverNormalizedSignal = -1.;
              if (signal[k1][k2][k3] > 0. && calib0[k1][kkk2][kkk3] > 0.) {
                GetRMSOverNormalizedSignal = signal[k1][k2][k3] / calib0[k1][kkk2][kkk3];
                if (k1 == 0) {
                  h_mapGetRMSOverNormalizedSignal_HB->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HB->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 1) {
                  h_mapGetRMSOverNormalizedSignal_HE->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HE->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 2) {
                  h_mapGetRMSOverNormalizedSignal_HO->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HO->Fill(double(k2plot), double(k3), 1.);
                }
                if (k1 == 3) {
                  h_mapGetRMSOverNormalizedSignal_HF->Fill(double(k2plot), double(k3), GetRMSOverNormalizedSignal);
                  h_mapGetRMSOverNormalizedSignal0_HF->Fill(double(k2plot), double(k3), 1.);
                }
              }
            }  //optimization
            ////////////////////////////////////////////////////////////////  for zcalib....C script:
            if (signal[k1][k2][k3] > 0.) {
              // ADC
              double adc = 0.;
              if (calib0[k1][kkk2][kkk3] > 0.)
                adc = calib0[k1][kkk2][kkk3];
              // Ratio
              double ratio = 2.;
              if (calib0[k1][kkk2][kkk3] > 0.)
                ratio = calib2[k1][kkk2][kkk3] / calib0[k1][kkk2][kkk3];
              // TSmax
              float calibtsmax = calibt[k1][kkk2][kkk3];
              // TSmean
              float calibtsmean = caliba[k1][kkk2][kkk3];
              // Width
              float calibwidth = calibw[k1][kkk2][kkk3];
              // CapIdErrors
              float calibcap = -100.;
              calibcap = calibcapiderror[k1][kkk2][kkk3];

              //                 HB:
              if (k1 == 0) {
                // ADC
                h_ADCCalib_HB->Fill(adc, 1.);
                h_ADCCalib1_HB->Fill(adc, 1.);
                if (adc < calibrADCHBMin_ || adc > calibrADCHBMax_)
                  h_mapADCCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HB->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HB->Fill(ratio, 1.);
                if (ratio < calibrRatioHBMin_ || ratio > calibrRatioHBMax_)
                  h_mapRatioCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HB->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HB->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHBMin_ || calibtsmax > calibrTSmaxHBMax_)
                    h_mapTSmaxCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HB->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HB->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHBMin_ || calibtsmean > calibrTSmeanHBMax_)
                    h_mapTSmeanCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HB->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HB->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHBMin_ || calibwidth > calibrWidthHBMax_)
                    h_mapWidthCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HB->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HB->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HB->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HE:
              if (k1 == 1) {
                // ADC
                h_ADCCalib_HE->Fill(adc, 1.);
                h_ADCCalib1_HE->Fill(adc, 1.);
                if (adc < calibrADCHEMin_ || adc > calibrADCHEMax_)
                  h_mapADCCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HE->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HE->Fill(ratio, 1.);
                if (ratio < calibrRatioHEMin_ || ratio > calibrRatioHEMax_)
                  h_mapRatioCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HE->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HE->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHEMin_ || calibtsmax > calibrTSmaxHEMax_)
                    h_mapTSmaxCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HE->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HE->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHEMin_ || calibtsmean > calibrTSmeanHEMax_)
                    h_mapTSmeanCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HE->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HE->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHEMin_ || calibwidth > calibrWidthHEMax_)
                    h_mapWidthCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HE->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HE->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HE->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HO:
              if (k1 == 2) {
                // ADC
                h_ADCCalib_HO->Fill(adc, 1.);
                h_ADCCalib1_HO->Fill(adc, 1.);
                if (adc < calibrADCHOMin_ || adc > calibrADCHOMax_)
                  h_mapADCCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HO->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HO->Fill(ratio, 1.);
                if (ratio < calibrRatioHOMin_ || ratio > calibrRatioHOMax_)
                  h_mapRatioCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HO->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HO->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHOMin_ || calibtsmax > calibrTSmaxHOMax_)
                    h_mapTSmaxCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HO->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HO->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHOMin_ || calibtsmean > calibrTSmeanHOMax_)
                    h_mapTSmeanCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HO->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HO->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHOMin_ || calibwidth > calibrWidthHOMax_)
                    h_mapWidthCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HO->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HO->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HO->Fill(double(k2plot), double(k3), 1.);
              }
              //                 HF:
              if (k1 == 3) {
                // ADC
                h_ADCCalib_HF->Fill(adc, 1.);
                h_ADCCalib1_HF->Fill(adc, 1.);
                if (adc < calibrADCHFMin_ || adc > calibrADCHFMax_)
                  h_mapADCCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                h_mapADCCalib_HF->Fill(double(k2plot), double(k3), adc);
                // Ratio
                h_RatioCalib_HF->Fill(ratio, 1.);
                if (ratio < calibrRatioHFMin_ || ratio > calibrRatioHFMax_)
                  h_mapRatioCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                h_mapRatioCalib_HF->Fill(double(k2plot), double(k3), ratio);
                // TSmax
                if (calibtsmax > -0.5) {
                  h_TSmaxCalib_HF->Fill(calibtsmax, 1.);
                  if (calibtsmax < calibrTSmaxHFMin_ || calibtsmax > calibrTSmaxHFMax_)
                    h_mapTSmaxCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmaxCalib_HF->Fill(double(k2plot), double(k3), calibtsmax);
                }
                // TSmean
                if (calibtsmean > -0.5) {
                  h_TSmeanCalib_HF->Fill(calibtsmean, 1.);
                  if (calibtsmean < calibrTSmeanHFMin_ || calibtsmean > calibrTSmeanHFMax_)
                    h_mapTSmeanCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapTSmeanCalib_HF->Fill(double(k2plot), double(k3), calibtsmean);
                }
                // Width
                if (calibwidth > -0.5) {
                  h_WidthCalib_HF->Fill(calibwidth, 1.);
                  if (calibwidth < calibrWidthHFMin_ || calibwidth > calibrWidthHFMax_)
                    h_mapWidthCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                  h_mapWidthCalib_HF->Fill(double(k2plot), double(k3), calibwidth);
                }
                // CapIdErrors
                if (calibcap > 0)
                  h_mapCapCalib047_HF->Fill(double(k2plot), double(k3), 1.);
                //
                h_map_HF->Fill(double(k2plot), double(k3), 1.);
              }
              //////////
            }  // if(signal[k1][k2][k3]>0.)
            //////////
          }  // k3
        }    // k2
      }      // k1

      /////

    }  //if(recordHistoes_&& studyCalibCellsHist_)

    ///////////////////////////////////////////////////
    if (recordNtuples_ && nevent50 < maxNeventsInNtuple_)
      myTree->Fill();
    //  if(recordNtuples_ && nevent < maxNeventsInNtuple_) myTree->Fill();

    ///////////////////////////////////////////////////
    if (++local_event % 100 == 0) {
      if (verbosity == -22)
        cout << "run " << Run << " processing events " << local_event << " ok, "
             << ", lumi " << lumi << ", numOfLaserEv " << numOfLaserEv << endl;
    }
  }  // bcn

  //EndAnalyzer
}

// ------------ method called once each job just before starting event loop  -----------
void CMTRawAnalyzer::beginJob() {
  if (verbosity > 0)
    cout << "========================   beignJob START   +++++++++++++++++++++++++++" << endl;
  hOutputFile = new TFile(fOutputFileName.c_str(), "RECREATE");
  ////////////////////////////////////////////////////////////////////////////////////////////////////////
  nnnnnn = 0;
  nnnnnnhbhe = 0;
  nnnnnnhbheqie11 = 0;
  nevent = 0;
  nevent50 = 0;
  counterhf = 0;
  counterhfqie10 = 0;
  counterho = 0;
  nnnnnn1 = 0;
  nnnnnn2 = 0;
  nnnnnn3 = 0;
  nnnnnn4 = 0;
  nnnnnn5 = 0;
  nnnnnn6 = 0;
  //  nnnnnn7= 0;
  //  nnnnnn8= 0;

  //////////////////////////////////////////////////////////////////////////////////    book histoes

  if (recordHistoes_) {
    //  ha2 = new TH2F("ha2"," ", neta, -41., 41., nphi, 0., bphi);

    h_errorGeneral = new TH1F("h_errorGeneral", " ", 5, 0., 5.);
    h_error1 = new TH1F("h_error1", " ", 5, 0., 5.);
    h_error2 = new TH1F("h_error2", " ", 5, 0., 5.);
    h_error3 = new TH1F("h_error3", " ", 5, 0., 5.);
    h_amplError = new TH1F("h_amplError", " ", 100, -2., 98.);
    h_amplFine = new TH1F("h_amplFine", " ", 100, -2., 98.);

    h_errorGeneral_HB = new TH1F("h_errorGeneral_HB", " ", 5, 0., 5.);
    h_error1_HB = new TH1F("h_error1_HB", " ", 5, 0., 5.);
    h_error2_HB = new TH1F("h_error2_HB", " ", 5, 0., 5.);
    h_error3_HB = new TH1F("h_error3_HB", " ", 5, 0., 5.);
    h_error4_HB = new TH1F("h_error4_HB", " ", 5, 0., 5.);
    h_error5_HB = new TH1F("h_error5_HB", " ", 5, 0., 5.);
    h_error6_HB = new TH1F("h_error6_HB", " ", 5, 0., 5.);
    h_error7_HB = new TH1F("h_error7_HB", " ", 5, 0., 5.);
    h_amplError_HB = new TH1F("h_amplError_HB", " ", 100, -2., 98.);
    h_amplFine_HB = new TH1F("h_amplFine_HB", " ", 100, -2., 98.);
    h_mapDepth1Error_HB = new TH2F("h_mapDepth1Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HB = new TH2F("h_mapDepth2Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HB = new TH2F("h_mapDepth3Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HB = new TH2F("h_mapDepth4Error_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HB = new TH1F("h_fiber0_HB", " ", 10, 0., 10.);
    h_fiber1_HB = new TH1F("h_fiber1_HB", " ", 10, 0., 10.);
    h_fiber2_HB = new TH1F("h_fiber2_HB", " ", 40, 0., 40.);
    h_repetedcapid_HB = new TH1F("h_repetedcapid_HB", " ", 5, 0., 5.);

    h_errorGeneral_HE = new TH1F("h_errorGeneral_HE", " ", 5, 0., 5.);
    h_error1_HE = new TH1F("h_error1_HE", " ", 5, 0., 5.);
    h_error2_HE = new TH1F("h_error2_HE", " ", 5, 0., 5.);
    h_error3_HE = new TH1F("h_error3_HE", " ", 5, 0., 5.);
    h_error4_HE = new TH1F("h_error4_HE", " ", 5, 0., 5.);
    h_error5_HE = new TH1F("h_error5_HE", " ", 5, 0., 5.);
    h_error6_HE = new TH1F("h_error6_HE", " ", 5, 0., 5.);
    h_error7_HE = new TH1F("h_error7_HE", " ", 5, 0., 5.);
    h_amplError_HE = new TH1F("h_amplError_HE", " ", 100, -2., 98.);
    h_amplFine_HE = new TH1F("h_amplFine_HE", " ", 100, -2., 98.);
    h_mapDepth1Error_HE = new TH2F("h_mapDepth1Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HE = new TH2F("h_mapDepth2Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HE = new TH2F("h_mapDepth3Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HE = new TH2F("h_mapDepth4Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Error_HE = new TH2F("h_mapDepth5Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Error_HE = new TH2F("h_mapDepth6Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Error_HE = new TH2F("h_mapDepth7Error_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HE = new TH1F("h_fiber0_HE", " ", 10, 0., 10.);
    h_fiber1_HE = new TH1F("h_fiber1_HE", " ", 10, 0., 10.);
    h_fiber2_HE = new TH1F("h_fiber2_HE", " ", 40, 0., 40.);
    h_repetedcapid_HE = new TH1F("h_repetedcapid_HE", " ", 5, 0., 5.);

    h_errorGeneral_HF = new TH1F("h_errorGeneral_HF", " ", 5, 0., 5.);
    h_error1_HF = new TH1F("h_error1_HF", " ", 5, 0., 5.);
    h_error2_HF = new TH1F("h_error2_HF", " ", 5, 0., 5.);
    h_error3_HF = new TH1F("h_error3_HF", " ", 5, 0., 5.);
    h_error4_HF = new TH1F("h_error4_HF", " ", 5, 0., 5.);
    h_error5_HF = new TH1F("h_error5_HF", " ", 5, 0., 5.);
    h_error6_HF = new TH1F("h_error6_HF", " ", 5, 0., 5.);
    h_error7_HF = new TH1F("h_error7_HF", " ", 5, 0., 5.);
    h_amplError_HF = new TH1F("h_amplError_HF", " ", 100, -2., 98.);
    h_amplFine_HF = new TH1F("h_amplFine_HF", " ", 100, -2., 98.);
    h_mapDepth1Error_HF = new TH2F("h_mapDepth1Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Error_HF = new TH2F("h_mapDepth2Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Error_HF = new TH2F("h_mapDepth3Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Error_HF = new TH2F("h_mapDepth4Error_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HF = new TH1F("h_fiber0_HF", " ", 10, 0., 10.);
    h_fiber1_HF = new TH1F("h_fiber1_HF", " ", 10, 0., 10.);
    h_fiber2_HF = new TH1F("h_fiber2_HF", " ", 40, 0., 40.);
    h_repetedcapid_HF = new TH1F("h_repetedcapid_HF", " ", 5, 0., 5.);

    h_errorGeneral_HO = new TH1F("h_errorGeneral_HO", " ", 5, 0., 5.);
    h_error1_HO = new TH1F("h_error1_HO", " ", 5, 0., 5.);
    h_error2_HO = new TH1F("h_error2_HO", " ", 5, 0., 5.);
    h_error3_HO = new TH1F("h_error3_HO", " ", 5, 0., 5.);
    h_error4_HO = new TH1F("h_error4_HO", " ", 5, 0., 5.);
    h_error5_HO = new TH1F("h_error5_HO", " ", 5, 0., 5.);
    h_error6_HO = new TH1F("h_error6_HO", " ", 5, 0., 5.);
    h_error7_HO = new TH1F("h_error7_HO", " ", 5, 0., 5.);
    h_amplError_HO = new TH1F("h_amplError_HO", " ", 100, -2., 98.);
    h_amplFine_HO = new TH1F("h_amplFine_HO", " ", 100, -2., 98.);
    h_mapDepth4Error_HO = new TH2F("h_mapDepth4Error_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_fiber0_HO = new TH1F("h_fiber0_HO", " ", 10, 0., 10.);
    h_fiber1_HO = new TH1F("h_fiber1_HO", " ", 10, 0., 10.);
    h_fiber2_HO = new TH1F("h_fiber2_HO", " ", 40, 0., 40.);
    h_repetedcapid_HO = new TH1F("h_repetedcapid_HO", " ", 5, 0., 5.);

    /////////////////////////////////////////////////////////////////////////////////////////////////             HB

    h_numberofhitsHBtest = new TH1F("h_numberofhitsHBtest", " ", 100, 0., 100.);
    h_AmplitudeHBtest = new TH1F("h_AmplitudeHBtest", " ", 100, 0., 10000.);
    h_AmplitudeHBtest1 = new TH1F("h_AmplitudeHBtest1", " ", 100, 0., 1000000.);
    h_AmplitudeHBtest6 = new TH1F("h_AmplitudeHBtest6", " ", 100, 0., 2000000.);
    h_totalAmplitudeHB = new TH1F("h_totalAmplitudeHB", " ", 100, 0., 3000000.);
    h_totalAmplitudeHBperEvent = new TH1F("h_totalAmplitudeHBperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmpl345Zoom_HB = new TH1F("h_ADCAmpl345Zoom_HB", " ", 100, 0., 400.);
    h_ADCAmpl345Zoom1_HB = new TH1F("h_ADCAmpl345Zoom1_HB", " ", 100, 0., 100.);
    h_ADCAmpl345_HB = new TH1F("h_ADCAmpl345_HB", " ", 100, 10., 3000.);

    h_AmplitudeHBrest = new TH1F("h_AmplitudeHBrest", " ", 100, 0., 10000.);
    h_AmplitudeHBrest1 = new TH1F("h_AmplitudeHBrest1", " ", 100, 0., 1000000.);
    h_AmplitudeHBrest6 = new TH1F("h_AmplitudeHBrest6", " ", 100, 0., 2000000.);

    h_ADCAmpl345_HBCapIdError = new TH1F("h_ADCAmpl345_HBCapIdError", " ", 100, 10., 3000.);
    h_ADCAmpl345_HBCapIdNoError = new TH1F("h_ADCAmpl345_HBCapIdNoError", " ", 100, 10., 3000.);
    h_ADCAmpl_HBCapIdError = new TH1F("h_ADCAmpl_HBCapIdError", " ", 100, 10., 3000.);
    h_ADCAmpl_HBCapIdNoError = new TH1F("h_ADCAmpl_HBCapIdNoError", " ", 100, 10., 3000.);

    h_ADCAmplZoom_HB = new TH1F("h_ADCAmplZoom_HB", " ", 100, 0., 400.);
    h_ADCAmplZoom1_HB = new TH1F("h_ADCAmplZoom1_HB", " ", 100, -20., 80.);
    h_ADCAmpl_HB = new TH1F("h_ADCAmpl_HB", " ", 100, 10., 5000.);
    h_mapDepth1ADCAmpl225_HB = new TH2F("h_mapDepth1ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HB = new TH2F("h_mapDepth2ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HB = new TH2F("h_mapDepth3ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HB = new TH2F("h_mapDepth4ADCAmpl225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HB = new TH2F("h_mapDepth1ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HB = new TH2F("h_mapDepth2ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HB = new TH2F("h_mapDepth3ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HB = new TH2F("h_mapDepth4ADCAmpl225Copy_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl_HB = new TH2F("h_mapDepth1ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HB = new TH2F("h_mapDepth2ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HB = new TH2F("h_mapDepth3ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HB = new TH2F("h_mapDepth4ADCAmpl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HB = new TH1F("h_TSmeanA_HB", " ", 100, -1., 11.);
    h_mapDepth1TSmeanA225_HB = new TH2F("h_mapDepth1TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HB = new TH2F("h_mapDepth2TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HB = new TH2F("h_mapDepth3TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HB = new TH2F("h_mapDepth4TSmeanA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HB = new TH2F("h_mapDepth1TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HB = new TH2F("h_mapDepth2TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HB = new TH2F("h_mapDepth3TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HB = new TH2F("h_mapDepth4TSmeanA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HB = new TH1F("h_TSmaxA_HB", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HB = new TH2F("h_mapDepth1TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HB = new TH2F("h_mapDepth2TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HB = new TH2F("h_mapDepth3TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HB = new TH2F("h_mapDepth4TSmaxA225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HB = new TH2F("h_mapDepth1TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HB = new TH2F("h_mapDepth2TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HB = new TH2F("h_mapDepth3TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HB = new TH2F("h_mapDepth4TSmaxA_HB", " ", neta, -41., 41., nphi, 0., bphi);
    // RMS:
    h_Amplitude_HB = new TH1F("h_Amplitude_HB", " ", 100, 0., 5.);
    h_mapDepth1Amplitude225_HB = new TH2F("h_mapDepth1Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HB = new TH2F("h_mapDepth2Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HB = new TH2F("h_mapDepth3Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HB = new TH2F("h_mapDepth4Amplitude225_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HB = new TH2F("h_mapDepth1Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HB = new TH2F("h_mapDepth2Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HB = new TH2F("h_mapDepth3Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HB = new TH2F("h_mapDepth4Amplitude_HB", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HB = new TH1F("h_Ampl_HB", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HB = new TH2F("h_mapDepth1Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HB = new TH2F("h_mapDepth2Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HB = new TH2F("h_mapDepth3Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HB = new TH2F("h_mapDepth4Ampl047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HB = new TH2F("h_mapDepth1Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HB = new TH2F("h_mapDepth2Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HB = new TH2F("h_mapDepth3Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HB = new TH2F("h_mapDepth4Ampl_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HB = new TH2F("h_mapDepth1AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HB = new TH2F("h_mapDepth2AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HB = new TH2F("h_mapDepth3AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HB = new TH2F("h_mapDepth4AmplE34_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HB = new TH2F("h_mapDepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HB = new TH2F("h_mapDepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HB = new TH2F("h_mapDepth3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HB = new TH2F("h_mapDepth4_HB", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////////////             HE

    // stuff regarding summed(total) Amplitude vs iEvent (histo-name is  h_totalAmplitudeHEperEvent)
    // to see from which event ALL channels are available(related to quality of the run)
    h_numberofhitsHEtest = new TH1F("h_numberofhitsHEtest", " ", 100, 0., 10000.);
    h_AmplitudeHEtest = new TH1F("h_AmplitudeHEtest", " ", 100, 0., 1000000.);
    h_AmplitudeHEtest1 = new TH1F("h_AmplitudeHEtest1", " ", 100, 0., 1000000.);
    h_AmplitudeHEtest6 = new TH1F("h_AmplitudeHEtest6", " ", 100, 0., 2000000.);
    h_totalAmplitudeHE = new TH1F("h_totalAmplitudeHE", " ", 100, 0., 10000000000.);
    h_totalAmplitudeHEperEvent = new TH1F("h_totalAmplitudeHEperEvent", " ", 1000, 1., 1001.);

    // Aijk Amplitude:
    h_ADCAmplZoom1_HE = new TH1F("h_ADCAmplZoom1_HE", " ", npfit, 0., anpfit);        // for amplmaxts 1TS w/ max
    h_ADCAmpl345Zoom1_HE = new TH1F("h_ADCAmpl345Zoom1_HE", " ", npfit, 0., anpfit);  // for ampl3ts 3TSs
    h_ADCAmpl345Zoom_HE = new TH1F("h_ADCAmpl345Zoom_HE", " ", npfit, 0., anpfit);    // for ampl 4TSs
    h_amplitudeaveragedbydepthes_HE =
        new TH1F("h_amplitudeaveragedbydepthes_HE", " ", npfit, 0., anpfit);  // for cross-check: A spectrum
    h_ndepthesperamplitudebins_HE =
        new TH1F("h_ndepthesperamplitudebins_HE", " ", 10, 0., 10.);  // for cross-check: ndepthes

    // Ampl12 4TSs to work with "ped-Gsel0" or "led-low-intensity" to clarify gain diff peak2-peak1
    h_mapADCAmplfirstpeak_HE =
        new TH2F("h_mapADCAmplfirstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplfirstpeak0_HE =
        new TH2F("h_mapADCAmplfirstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplsecondpeak_HE =
        new TH2F("h_mapADCAmplsecondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max
    h_mapADCAmplsecondpeak0_HE =
        new TH2F("h_mapADCAmplsecondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for amplmaxts 1TS w/ max

    h_mapADCAmpl11firstpeak_HE =
        new TH2F("h_mapADCAmpl11firstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11firstpeak0_HE =
        new TH2F("h_mapADCAmpl11firstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11secondpeak_HE =
        new TH2F("h_mapADCAmpl11secondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs
    h_mapADCAmpl11secondpeak0_HE =
        new TH2F("h_mapADCAmpl11secondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl3ts 3TSs

    h_mapADCAmpl12firstpeak_HE =
        new TH2F("h_mapADCAmpl12firstpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12firstpeak0_HE =
        new TH2F("h_mapADCAmpl12firstpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12secondpeak_HE =
        new TH2F("h_mapADCAmpl12secondpeak_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs
    h_mapADCAmpl12secondpeak0_HE =
        new TH2F("h_mapADCAmpl12secondpeak0_HE", " ", neta, -41., 41., nphi, 0., bphi);  // for ampl 4TSs

    // Ampl12 4TSs to work with "ped-Gsel0" or "led-low-intensity" to clarify gain diff peak2-peak1  fit results:
    h_gsmdifferencefit1_HE = new TH1F("h_gsmdifferencefit1_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit2_HE = new TH1F("h_gsmdifferencefit2_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit3_HE = new TH1F("h_gsmdifferencefit3_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit4_HE = new TH1F("h_gsmdifferencefit4_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit5_HE = new TH1F("h_gsmdifferencefit5_HE", " ", 80, 20., 60.);
    h_gsmdifferencefit6_HE = new TH1F("h_gsmdifferencefit6_HE", " ", 80, 20., 60.);

    // Aijk Amplitude:
    h_ADCAmpl_HE = new TH1F("h_ADCAmpl_HE", " ", 200, 0., 2000000.);
    h_ADCAmplrest_HE = new TH1F("h_ADCAmplrest_HE", " ", 100, 0., 500.);
    h_ADCAmplrest1_HE = new TH1F("h_ADCAmplrest1_HE", " ", 100, 0., 100.);
    h_ADCAmplrest6_HE = new TH1F("h_ADCAmplrest6_HE", " ", 100, 0., 10000.);

    h_ADCAmpl345_HE = new TH1F("h_ADCAmpl345_HE", " ", 70, 0., 700000.);

    // SiPM corrections:
    h_corrforxaMAIN_HE = new TH1F("h_corrforxaMAIN_HE", " ", 70, 0., 700000.);
    h_corrforxaMAIN0_HE = new TH1F("h_corrforxaMAIN0_HE", " ", 70, 0., 700000.);
    h_corrforxaADDI_HE = new TH1F("h_corrforxaADDI_HE", " ", 70, 0., 700000.);
    h_corrforxaADDI0_HE = new TH1F("h_corrforxaADDI0_HE", " ", 70, 0., 700000.);

    h_mapDepth1ADCAmpl225_HE = new TH2F("h_mapDepth1ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HE = new TH2F("h_mapDepth2ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HE = new TH2F("h_mapDepth3ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HE = new TH2F("h_mapDepth4ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl225_HE = new TH2F("h_mapDepth5ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl225_HE = new TH2F("h_mapDepth6ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl225_HE = new TH2F("h_mapDepth7ADCAmpl225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HE = new TH2F("h_mapDepth1ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HE = new TH2F("h_mapDepth2ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HE = new TH2F("h_mapDepth3ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HE = new TH2F("h_mapDepth4ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl225Copy_HE = new TH2F("h_mapDepth5ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl225Copy_HE = new TH2F("h_mapDepth6ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl225Copy_HE = new TH2F("h_mapDepth7ADCAmpl225Copy_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1ADCAmpl_HE = new TH2F("h_mapDepth1ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HE = new TH2F("h_mapDepth2ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HE = new TH2F("h_mapDepth3ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HE = new TH2F("h_mapDepth4ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl_HE = new TH2F("h_mapDepth5ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl_HE = new TH2F("h_mapDepth6ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl_HE = new TH2F("h_mapDepth7ADCAmpl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmplSiPM_HE = new TH2F("h_mapDepth1ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmplSiPM_HE = new TH2F("h_mapDepth2ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmplSiPM_HE = new TH2F("h_mapDepth3ADCAmplSiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_TSmeanA_HE = new TH1F("h_TSmeanA_HE", " ", 100, -2., 8.);
    h_mapDepth1TSmeanA225_HE = new TH2F("h_mapDepth1TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HE = new TH2F("h_mapDepth2TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HE = new TH2F("h_mapDepth3TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HE = new TH2F("h_mapDepth4TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmeanA225_HE = new TH2F("h_mapDepth5TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmeanA225_HE = new TH2F("h_mapDepth6TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmeanA225_HE = new TH2F("h_mapDepth7TSmeanA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HE = new TH2F("h_mapDepth1TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HE = new TH2F("h_mapDepth2TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HE = new TH2F("h_mapDepth3TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HE = new TH2F("h_mapDepth4TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmeanA_HE = new TH2F("h_mapDepth5TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmeanA_HE = new TH2F("h_mapDepth6TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmeanA_HE = new TH2F("h_mapDepth7TSmeanA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HE = new TH1F("h_TSmaxA_HE", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HE = new TH2F("h_mapDepth1TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HE = new TH2F("h_mapDepth2TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HE = new TH2F("h_mapDepth3TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HE = new TH2F("h_mapDepth4TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmaxA225_HE = new TH2F("h_mapDepth5TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmaxA225_HE = new TH2F("h_mapDepth6TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmaxA225_HE = new TH2F("h_mapDepth7TSmaxA225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HE = new TH2F("h_mapDepth1TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HE = new TH2F("h_mapDepth2TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HE = new TH2F("h_mapDepth3TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HE = new TH2F("h_mapDepth4TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5TSmaxA_HE = new TH2F("h_mapDepth5TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6TSmaxA_HE = new TH2F("h_mapDepth6TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7TSmaxA_HE = new TH2F("h_mapDepth7TSmaxA_HE", " ", neta, -41., 41., nphi, 0., bphi);
    // RMS:
    h_Amplitude_HE = new TH1F("h_Amplitude_HE", " ", 100, 0., 5.5);
    h_mapDepth1Amplitude225_HE = new TH2F("h_mapDepth1Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HE = new TH2F("h_mapDepth2Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HE = new TH2F("h_mapDepth3Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HE = new TH2F("h_mapDepth4Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Amplitude225_HE = new TH2F("h_mapDepth5Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Amplitude225_HE = new TH2F("h_mapDepth6Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Amplitude225_HE = new TH2F("h_mapDepth7Amplitude225_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HE = new TH2F("h_mapDepth1Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HE = new TH2F("h_mapDepth2Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HE = new TH2F("h_mapDepth3Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HE = new TH2F("h_mapDepth4Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Amplitude_HE = new TH2F("h_mapDepth5Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Amplitude_HE = new TH2F("h_mapDepth6Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Amplitude_HE = new TH2F("h_mapDepth7Amplitude_HE", " ", neta, -41., 41., nphi, 0., bphi);

    // Ratio:
    h_Ampl_HE = new TH1F("h_Ampl_HE", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HE = new TH2F("h_mapDepth1Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HE = new TH2F("h_mapDepth2Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HE = new TH2F("h_mapDepth3Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HE = new TH2F("h_mapDepth4Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Ampl047_HE = new TH2F("h_mapDepth5Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Ampl047_HE = new TH2F("h_mapDepth6Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Ampl047_HE = new TH2F("h_mapDepth7Ampl047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HE = new TH2F("h_mapDepth1Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HE = new TH2F("h_mapDepth2Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HE = new TH2F("h_mapDepth3Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HE = new TH2F("h_mapDepth4Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5Ampl_HE = new TH2F("h_mapDepth5Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6Ampl_HE = new TH2F("h_mapDepth6Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7Ampl_HE = new TH2F("h_mapDepth7Ampl_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HE = new TH2F("h_mapDepth1AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HE = new TH2F("h_mapDepth2AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HE = new TH2F("h_mapDepth3AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HE = new TH2F("h_mapDepth4AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5AmplE34_HE = new TH2F("h_mapDepth5AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6AmplE34_HE = new TH2F("h_mapDepth6AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7AmplE34_HE = new TH2F("h_mapDepth7AmplE34_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HE = new TH2F("h_mapDepth1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HE = new TH2F("h_mapDepth2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HE = new TH2F("h_mapDepth3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HE = new TH2F("h_mapDepth4_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5_HE = new TH2F("h_mapDepth5_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6_HE = new TH2F("h_mapDepth6_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7_HE = new TH2F("h_mapDepth7_HE", " ", neta, -41., 41., nphi, 0., bphi);
    ///////////////////////////////////////////////////////////////////////////////////////////////////  IterativeMethodCalibrationGroup
    h_mapenophinorm_HE1 = new TH2F("h_mapenophinorm_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE2 = new TH2F("h_mapenophinorm_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE3 = new TH2F("h_mapenophinorm_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE4 = new TH2F("h_mapenophinorm_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE5 = new TH2F("h_mapenophinorm_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE6 = new TH2F("h_mapenophinorm_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm_HE7 = new TH2F("h_mapenophinorm_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE1 = new TH2F("h_mapenophinorm2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE2 = new TH2F("h_mapenophinorm2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE3 = new TH2F("h_mapenophinorm2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE4 = new TH2F("h_mapenophinorm2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE5 = new TH2F("h_mapenophinorm2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE6 = new TH2F("h_mapenophinorm2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapenophinorm2_HE7 = new TH2F("h_mapenophinorm2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    h_maprphinorm_HE1 = new TH2F("h_maprphinorm_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE2 = new TH2F("h_maprphinorm_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE3 = new TH2F("h_maprphinorm_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE4 = new TH2F("h_maprphinorm_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE5 = new TH2F("h_maprphinorm_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE6 = new TH2F("h_maprphinorm_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm_HE7 = new TH2F("h_maprphinorm_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE1 = new TH2F("h_maprphinorm2_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE2 = new TH2F("h_maprphinorm2_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE3 = new TH2F("h_maprphinorm2_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE4 = new TH2F("h_maprphinorm2_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE5 = new TH2F("h_maprphinorm2_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE6 = new TH2F("h_maprphinorm2_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm2_HE7 = new TH2F("h_maprphinorm2_HE7", " ", neta, -41., 41., nphi, 0., bphi);

    h_maprphinorm0_HE1 = new TH2F("h_maprphinorm0_HE1", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE2 = new TH2F("h_maprphinorm0_HE2", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE3 = new TH2F("h_maprphinorm0_HE3", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE4 = new TH2F("h_maprphinorm0_HE4", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE5 = new TH2F("h_maprphinorm0_HE5", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE6 = new TH2F("h_maprphinorm0_HE6", " ", neta, -41., 41., nphi, 0., bphi);
    h_maprphinorm0_HE7 = new TH2F("h_maprphinorm0_HE7", " ", neta, -41., 41., nphi, 0., bphi);
    ///////////////////////////////////////////////////////////////////////////////////////////////////  raddam:
    // RADDAM:
    //    if(flagLaserRaddam_ == 1 ) {
    //    }
    int min80 = -100.;
    int max80 = 9000.;
    // fill for each digi (=each event, each channel)
    h_mapDepth1RADDAM_HE = new TH2F("h_mapDepth1RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2RADDAM_HE = new TH2F("h_mapDepth2RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3RADDAM_HE = new TH2F("h_mapDepth3RADDAM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1RADDAM0_HE = new TH2F("h_mapDepth1RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2RADDAM0_HE = new TH2F("h_mapDepth2RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3RADDAM0_HE = new TH2F("h_mapDepth3RADDAM0_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_sigLayer1RADDAM_HE = new TH1F("h_sigLayer1RADDAM_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM_HE = new TH1F("h_sigLayer2RADDAM_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM0_HE = new TH1F("h_sigLayer1RADDAM0_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM0_HE = new TH1F("h_sigLayer2RADDAM0_HE", " ", neta, -41., 41.);
    h_AamplitudewithPedSubtr_RADDAM_HE = new TH1F("h_AamplitudewithPedSubtr_RADDAM_HE", " ", 100, min80, max80);
    h_AamplitudewithPedSubtr_RADDAM_HEzoom0 =
        new TH1F("h_AamplitudewithPedSubtr_RADDAM_HEzoom0", " ", 100, min80, 4000.);
    h_AamplitudewithPedSubtr_RADDAM_HEzoom1 =
        new TH1F("h_AamplitudewithPedSubtr_RADDAM_HEzoom1", " ", 100, min80, 1000.);
    h_mapDepth3RADDAM16_HE = new TH1F("h_mapDepth3RADDAM16_HE", " ", 100, min80, max80);
    h_A_Depth1RADDAM_HE = new TH1F("h_A_Depth1RADDAM_HE", " ", 100, min80, max80);
    h_A_Depth2RADDAM_HE = new TH1F("h_A_Depth2RADDAM_HE", " ", 100, min80, max80);
    h_A_Depth3RADDAM_HE = new TH1F("h_A_Depth3RADDAM_HE", " ", 100, min80, max80);
    int min90 = 0.;
    int max90 = 5000.;
    h_sumphiEta16Depth3RADDAM_HED2 = new TH1F("h_sumphiEta16Depth3RADDAM_HED2", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2 = new TH1F("h_Eta16Depth3RADDAM_HED2", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2 = new TH1F("h_NphiForEta16Depth3RADDAM_HED2", " ", 100, 0, 100.);
    h_sumphiEta16Depth3RADDAM_HED2P = new TH1F("h_sumphiEta16Depth3RADDAM_HED2P", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2P = new TH1F("h_Eta16Depth3RADDAM_HED2P", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2P = new TH1F("h_NphiForEta16Depth3RADDAM_HED2P", " ", 100, 0, 100.);
    h_sumphiEta16Depth3RADDAM_HED2ALL = new TH1F("h_sumphiEta16Depth3RADDAM_HED2ALL", " ", 100, min90, 70. * max90);
    h_Eta16Depth3RADDAM_HED2ALL = new TH1F("h_Eta16Depth3RADDAM_HED2ALL", " ", 100, min90, max90);
    h_NphiForEta16Depth3RADDAM_HED2ALL = new TH1F("h_NphiForEta16Depth3RADDAM_HED2ALL", " ", 100, 0, 100.);
    h_sigLayer1RADDAM5_HE = new TH1F("h_sigLayer1RADDAM5_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM5_HE = new TH1F("h_sigLayer2RADDAM5_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM6_HE = new TH1F("h_sigLayer1RADDAM6_HE", " ", neta, -41., 41.);
    h_sigLayer2RADDAM6_HE = new TH1F("h_sigLayer2RADDAM6_HE", " ", neta, -41., 41.);
    h_sigLayer1RADDAM5_HED2 = new TH1F("h_sigLayer1RADDAM5_HED2", " ", neta, -41., 41.);
    h_sigLayer2RADDAM5_HED2 = new TH1F("h_sigLayer2RADDAM5_HED2", " ", neta, -41., 41.);
    h_sigLayer1RADDAM6_HED2 = new TH1F("h_sigLayer1RADDAM6_HED2", " ", neta, -41., 41.);
    h_sigLayer2RADDAM6_HED2 = new TH1F("h_sigLayer2RADDAM6_HED2", " ", neta, -41., 41.);

    h_numberofhitsHFtest = new TH1F("h_numberofhitsHFtest", " ", 100, 0., 30000.);
    h_AmplitudeHFtest = new TH1F("h_AmplitudeHFtest", " ", 100, 0., 300000.);
    h_totalAmplitudeHF = new TH1F("h_totalAmplitudeHF", " ", 100, 0., 100000000000.);
    h_totalAmplitudeHFperEvent = new TH1F("h_totalAmplitudeHFperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmplZoom1_HF = new TH1F("h_ADCAmplZoom1_HF", " ", 100, 0., 1000000.);
    h_ADCAmpl_HF = new TH1F("h_ADCAmpl_HF", " ", 250, 0., 500000.);
    h_ADCAmplrest1_HF = new TH1F("h_ADCAmplrest1_HF", " ", 100, 0., 1000.);
    h_ADCAmplrest6_HF = new TH1F("h_ADCAmplrest6_HF", " ", 100, 0., 10000.);

    h_mapDepth1ADCAmpl225_HF = new TH2F("h_mapDepth1ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225_HF = new TH2F("h_mapDepth2ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl225Copy_HF = new TH2F("h_mapDepth1ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl225Copy_HF = new TH2F("h_mapDepth2ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl_HF = new TH2F("h_mapDepth1ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl_HF = new TH2F("h_mapDepth2ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225_HF = new TH2F("h_mapDepth3ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225_HF = new TH2F("h_mapDepth4ADCAmpl225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl225Copy_HF = new TH2F("h_mapDepth3ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HF = new TH2F("h_mapDepth4ADCAmpl225Copy_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl_HF = new TH2F("h_mapDepth3ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HF = new TH2F("h_mapDepth4ADCAmpl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HF = new TH1F("h_TSmeanA_HF", " ", 100, -1., 11.);
    h_mapDepth1TSmeanA225_HF = new TH2F("h_mapDepth1TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA225_HF = new TH2F("h_mapDepth2TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmeanA_HF = new TH2F("h_mapDepth1TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmeanA_HF = new TH2F("h_mapDepth2TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA225_HF = new TH2F("h_mapDepth3TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA225_HF = new TH2F("h_mapDepth4TSmeanA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmeanA_HF = new TH2F("h_mapDepth3TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HF = new TH2F("h_mapDepth4TSmeanA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HF = new TH1F("h_Amplitude_HF", " ", 100, 0., 5.);
    h_TSmaxA_HF = new TH1F("h_TSmaxA_HF", " ", 100, -1., 11.);
    h_mapDepth1TSmaxA225_HF = new TH2F("h_mapDepth1TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA225_HF = new TH2F("h_mapDepth2TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1TSmaxA_HF = new TH2F("h_mapDepth1TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2TSmaxA_HF = new TH2F("h_mapDepth2TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA225_HF = new TH2F("h_mapDepth3TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA225_HF = new TH2F("h_mapDepth4TSmaxA225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3TSmaxA_HF = new TH2F("h_mapDepth3TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HF = new TH2F("h_mapDepth4TSmaxA_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HF = new TH1F("h_Amplitude_HF", " ", 100, 0., 5.);
    h_mapDepth1Amplitude225_HF = new TH2F("h_mapDepth1Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude225_HF = new TH2F("h_mapDepth2Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Amplitude_HF = new TH2F("h_mapDepth1Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Amplitude_HF = new TH2F("h_mapDepth2Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude225_HF = new TH2F("h_mapDepth3Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude225_HF = new TH2F("h_mapDepth4Amplitude225_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Amplitude_HF = new TH2F("h_mapDepth3Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HF = new TH2F("h_mapDepth4Amplitude_HF", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HF = new TH1F("h_Ampl_HF", " ", 100, 0., 1.1);
    h_mapDepth1Ampl047_HF = new TH2F("h_mapDepth1Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl047_HF = new TH2F("h_mapDepth2Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ampl_HF = new TH2F("h_mapDepth1Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ampl_HF = new TH2F("h_mapDepth2Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1AmplE34_HF = new TH2F("h_mapDepth1AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2AmplE34_HF = new TH2F("h_mapDepth2AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1_HF = new TH2F("h_mapDepth1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2_HF = new TH2F("h_mapDepth2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl047_HF = new TH2F("h_mapDepth3Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl047_HF = new TH2F("h_mapDepth4Ampl047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ampl_HF = new TH2F("h_mapDepth3Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HF = new TH2F("h_mapDepth4Ampl_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3AmplE34_HF = new TH2F("h_mapDepth3AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HF = new TH2F("h_mapDepth4AmplE34_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3_HF = new TH2F("h_mapDepth3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HF = new TH2F("h_mapDepth4_HF", " ", neta, -41., 41., nphi, 0., bphi);

    ////////////////////////////////////////////////////////////////////////////////////////////////                  HO
    h_numberofhitsHOtest = new TH1F("h_numberofhitsHOtest", " ", 100, 0., 30000.);
    h_AmplitudeHOtest = new TH1F("h_AmplitudeHOtest", " ", 100, 0., 300000.);
    h_totalAmplitudeHO = new TH1F("h_totalAmplitudeHO", " ", 100, 0., 100000000.);
    h_totalAmplitudeHOperEvent = new TH1F("h_totalAmplitudeHOperEvent", " ", 1000, 1., 1001.);
    // fullAmplitude:
    h_ADCAmpl_HO = new TH1F("h_ADCAmpl_HO", " ", 100, 0., 7000.);
    h_ADCAmplrest1_HO = new TH1F("h_ADCAmplrest1_HO", " ", 100, 0., 150.);
    h_ADCAmplrest6_HO = new TH1F("h_ADCAmplrest6_HO", " ", 100, 0., 500.);

    h_ADCAmplZoom1_HO = new TH1F("h_ADCAmplZoom1_HO", " ", 100, -20., 280.);
    h_ADCAmpl_HO_copy = new TH1F("h_ADCAmpl_HO_copy", " ", 100, 0., 30000.);
    h_mapDepth4ADCAmpl225_HO = new TH2F("h_mapDepth4ADCAmpl225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl225Copy_HO = new TH2F("h_mapDepth4ADCAmpl225Copy_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl_HO = new TH2F("h_mapDepth4ADCAmpl_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanA_HO = new TH1F("h_TSmeanA_HO", " ", 100, 0., 10.);
    h_mapDepth4TSmeanA225_HO = new TH2F("h_mapDepth4TSmeanA225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmeanA_HO = new TH2F("h_mapDepth4TSmeanA_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxA_HO = new TH1F("h_TSmaxA_HO", " ", 100, 0., 10.);
    h_mapDepth4TSmaxA225_HO = new TH2F("h_mapDepth4TSmaxA225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4TSmaxA_HO = new TH2F("h_mapDepth4TSmaxA_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_Amplitude_HO = new TH1F("h_Amplitude_HO", " ", 100, 0., 5.);
    h_mapDepth4Amplitude225_HO = new TH2F("h_mapDepth4Amplitude225_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Amplitude_HO = new TH2F("h_mapDepth4Amplitude_HO", " ", neta, -41., 41., nphi, 0., bphi);
    // Ratio:
    h_Ampl_HO = new TH1F("h_Ampl_HO", " ", 100, 0., 1.1);
    h_mapDepth4Ampl047_HO = new TH2F("h_mapDepth4Ampl047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ampl_HO = new TH2F("h_mapDepth4Ampl_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4AmplE34_HO = new TH2F("h_mapDepth4AmplE34_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4_HO = new TH2F("h_mapDepth4_HO", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////
    int baP = 4000;
    float baR = 0.;
    float baR2 = baP;
    h_bcnnbadchannels_depth1_HB = new TH1F("h_bcnnbadchannels_depth1_HB", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HB = new TH1F("h_bcnnbadchannels_depth2_HB", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth1_HE = new TH1F("h_bcnnbadchannels_depth1_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HE = new TH1F("h_bcnnbadchannels_depth2_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth3_HE = new TH1F("h_bcnnbadchannels_depth3_HE", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth4_HO = new TH1F("h_bcnnbadchannels_depth4_HO", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth1_HF = new TH1F("h_bcnnbadchannels_depth1_HF", " ", baP, baR, baR2);
    h_bcnnbadchannels_depth2_HF = new TH1F("h_bcnnbadchannels_depth2_HF", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HB = new TH1F("h_bcnbadrate0_depth1_HB", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HB = new TH1F("h_bcnbadrate0_depth2_HB", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HE = new TH1F("h_bcnbadrate0_depth1_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HE = new TH1F("h_bcnbadrate0_depth2_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth3_HE = new TH1F("h_bcnbadrate0_depth3_HE", " ", baP, baR, baR2);
    h_bcnbadrate0_depth4_HO = new TH1F("h_bcnbadrate0_depth4_HO", " ", baP, baR, baR2);
    h_bcnbadrate0_depth1_HF = new TH1F("h_bcnbadrate0_depth1_HF", " ", baP, baR, baR2);
    h_bcnbadrate0_depth2_HF = new TH1F("h_bcnbadrate0_depth2_HF", " ", baP, baR, baR2);

    h_bcnvsamplitude_HB = new TH1F("h_bcnvsamplitude_HB", " ", baP, baR, baR2);
    h_bcnvsamplitude_HE = new TH1F("h_bcnvsamplitude_HE", " ", baP, baR, baR2);
    h_bcnvsamplitude_HF = new TH1F("h_bcnvsamplitude_HF", " ", baP, baR, baR2);
    h_bcnvsamplitude_HO = new TH1F("h_bcnvsamplitude_HO", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HB = new TH1F("h_bcnvsamplitude0_HB", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HE = new TH1F("h_bcnvsamplitude0_HE", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HF = new TH1F("h_bcnvsamplitude0_HF", " ", baP, baR, baR2);
    h_bcnvsamplitude0_HO = new TH1F("h_bcnvsamplitude0_HO", " ", baP, baR, baR2);

    int zaP = 1000;
    float zaR = 10000000.;
    float zaR2 = 50000000.;
    h_orbitNumvsamplitude_HB = new TH1F("h_orbitNumvsamplitude_HB", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HE = new TH1F("h_orbitNumvsamplitude_HE", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HF = new TH1F("h_orbitNumvsamplitude_HF", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude_HO = new TH1F("h_orbitNumvsamplitude_HO", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HB = new TH1F("h_orbitNumvsamplitude0_HB", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HE = new TH1F("h_orbitNumvsamplitude0_HE", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HF = new TH1F("h_orbitNumvsamplitude0_HF", " ", zaP, zaR, zaR2);
    h_orbitNumvsamplitude0_HO = new TH1F("h_orbitNumvsamplitude0_HO", " ", zaP, zaR, zaR2);

    h_2DsumADCAmplEtaPhiLs0 =
        new TH2F("h_2DsumADCAmplEtaPhiLs0", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs1 =
        new TH2F("h_2DsumADCAmplEtaPhiLs1", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs2 =
        new TH2F("h_2DsumADCAmplEtaPhiLs2", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs3 =
        new TH2F("h_2DsumADCAmplEtaPhiLs3", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);

    h_2DsumADCAmplEtaPhiLs00 =
        new TH2F("h_2DsumADCAmplEtaPhiLs00", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs10 =
        new TH2F("h_2DsumADCAmplEtaPhiLs10", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs20 =
        new TH2F("h_2DsumADCAmplEtaPhiLs20", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);
    h_2DsumADCAmplEtaPhiLs30 =
        new TH2F("h_2DsumADCAmplEtaPhiLs30", " ", nlsminmax, alsmin, blsmax, znphi * zneta, 1., znphi * zneta + 1.);

    h_sumADCAmplEtaPhiLs = new TH1F("h_sumADCAmplEtaPhiLs", " ", 1000, 0., 14000.);
    h_sumADCAmplEtaPhiLs_bbbc = new TH1F("h_sumADCAmplEtaPhiLs_bbbc", " ", 1000, 0., 300000.);
    h_sumADCAmplEtaPhiLs_bbb1 = new TH1F("h_sumADCAmplEtaPhiLs_bbb1", " ", 100, 0., 3000.);
    h_sumADCAmplEtaPhiLs_lscounterM1 = new TH1F("h_sumADCAmplEtaPhiLs_lscounterM1", " ", 600, 1., 601.);
    h_sumADCAmplEtaPhiLs_ietaphi = new TH1F("h_sumADCAmplEtaPhiLs_ietaphi", " ", 400, 0., 400.);
    h_sumADCAmplEtaPhiLs_lscounterM1orbitNum = new TH1F("h_sumADCAmplEtaPhiLs_lscounterM1orbitNum", " ", 600, 1., 601.);
    h_sumADCAmplEtaPhiLs_orbitNum = new TH1F("h_sumADCAmplEtaPhiLs_orbitNum", " ", 1000, 25000000., 40000000.);

    // for LS :

    // for LS binning:
    int bac = howmanybinsonplots_;
    //  int bac= 15;
    float bac2 = bac + 1.;
    // bac,         1.,     bac2  );

    h_nbadchannels_depth1_HB = new TH1F("h_nbadchannels_depth1_HB", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HB = new TH1F("h_runnbadchannels_depth1_HB", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HB = new TH1F("h_runnbadchannelsC_depth1_HB", " ", bac, 1., bac2);
    h_runbadrate_depth1_HB = new TH1F("h_runbadrate_depth1_HB", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HB = new TH1F("h_runbadrateC_depth1_HB", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HB = new TH1F("h_runbadrate0_depth1_HB", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HB = new TH1F("h_nbadchannels_depth2_HB", " ", 100, 1., 501.);
    h_runnbadchannels_depth2_HB = new TH1F("h_runnbadchannels_depth2_HB", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HB = new TH1F("h_runnbadchannelsC_depth2_HB", " ", bac, 1., bac2);
    h_runbadrate_depth2_HB = new TH1F("h_runbadrate_depth2_HB", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HB = new TH1F("h_runbadrateC_depth2_HB", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HB = new TH1F("h_runbadrate0_depth2_HB", " ", bac, 1., bac2);

    h_nbadchannels_depth1_HE = new TH1F("h_nbadchannels_depth1_HE", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HE = new TH1F("h_runnbadchannels_depth1_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HE = new TH1F("h_runnbadchannelsC_depth1_HE", " ", bac, 1., bac2);
    h_runbadrate_depth1_HE = new TH1F("h_runbadrate_depth1_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HE = new TH1F("h_runbadrateC_depth1_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HE = new TH1F("h_runbadrate0_depth1_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HE = new TH1F("h_nbadchannels_depth2_HE", " ", 100, 1., 3001.);
    h_runnbadchannels_depth2_HE = new TH1F("h_runnbadchannels_depth2_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HE = new TH1F("h_runnbadchannelsC_depth2_HE", " ", bac, 1., bac2);
    h_runbadrate_depth2_HE = new TH1F("h_runbadrate_depth2_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HE = new TH1F("h_runbadrateC_depth2_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HE = new TH1F("h_runbadrate0_depth2_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth3_HE = new TH1F("h_nbadchannels_depth3_HE", " ", 100, 1., 501.);
    h_runnbadchannels_depth3_HE = new TH1F("h_runnbadchannels_depth3_HE", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth3_HE = new TH1F("h_runnbadchannelsC_depth3_HE", " ", bac, 1., bac2);
    h_runbadrate_depth3_HE = new TH1F("h_runbadrate_depth3_HE", " ", bac, 1., bac2);
    h_runbadrateC_depth3_HE = new TH1F("h_runbadrateC_depth3_HE", " ", bac, 1., bac2);
    h_runbadrate0_depth3_HE = new TH1F("h_runbadrate0_depth3_HE", " ", bac, 1., bac2);

    h_nbadchannels_depth1_HF = new TH1F("h_nbadchannels_depth1_HF", " ", 100, 1., 3001.);
    h_runnbadchannels_depth1_HF = new TH1F("h_runnbadchannels_depth1_HF", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth1_HF = new TH1F("h_runnbadchannelsC_depth1_HF", " ", bac, 1., bac2);
    h_runbadrate_depth1_HF = new TH1F("h_runbadrate_depth1_HF", " ", bac, 1., bac2);
    h_runbadrateC_depth1_HF = new TH1F("h_runbadrateC_depth1_HF", " ", bac, 1., bac2);
    h_runbadrate0_depth1_HF = new TH1F("h_runbadrate0_depth1_HF", " ", bac, 1., bac2);

    h_nbadchannels_depth2_HF = new TH1F("h_nbadchannels_depth2_HF", " ", 100, 1., 501.);
    h_runnbadchannels_depth2_HF = new TH1F("h_runnbadchannels_depth2_HF", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth2_HF = new TH1F("h_runnbadchannelsC_depth2_HF", " ", bac, 1., bac2);
    h_runbadrate_depth2_HF = new TH1F("h_runbadrate_depth2_HF", " ", bac, 1., bac2);
    h_runbadrateC_depth2_HF = new TH1F("h_runbadrateC_depth2_HF", " ", bac, 1., bac2);
    h_runbadrate0_depth2_HF = new TH1F("h_runbadrate0_depth2_HF", " ", bac, 1., bac2);

    h_nbadchannels_depth4_HO = new TH1F("h_nbadchannels_depth4_HO", " ", 100, 1., 3001.);
    h_runnbadchannels_depth4_HO = new TH1F("h_runnbadchannels_depth4_HO", " ", bac, 1., bac2);
    h_runnbadchannelsC_depth4_HO = new TH1F("h_runnbadchannelsC_depth4_HO", " ", bac, 1., bac2);
    h_runbadrate_depth4_HO = new TH1F("h_runbadrate_depth4_HO", " ", bac, 1., bac2);
    h_runbadrateC_depth4_HO = new TH1F("h_runbadrateC_depth4_HO", " ", bac, 1., bac2);
    h_runbadrate0_depth4_HO = new TH1F("h_runbadrate0_depth4_HO", " ", bac, 1., bac2);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    h_FullSignal3D_HB = new TH2F("h_FullSignal3D_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HB = new TH2F("h_FullSignal3D0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HE = new TH2F("h_FullSignal3D_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HE = new TH2F("h_FullSignal3D0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HO = new TH2F("h_FullSignal3D_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HO = new TH2F("h_FullSignal3D0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D_HF = new TH2F("h_FullSignal3D_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_FullSignal3D0_HF = new TH2F("h_FullSignal3D0_HF", " ", neta, -41., 41., nphi, 0., bphi);

    //////////////////////////////////////////////////////////////////////////////////////////////////
    h_ADCCalib_HB = new TH1F("h_ADCCalib_HB", " ", 100, 10., 10000.);
    h_ADCCalib1_HB = new TH1F("h_ADCCalib1_HB", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HB = new TH2F("h_mapADCCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HB = new TH2F("h_mapADCCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HB = new TH1F("h_RatioCalib_HB", " ", 100, 0., 1.);
    h_mapRatioCalib047_HB = new TH2F("h_mapRatioCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HB = new TH2F("h_mapRatioCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HB = new TH1F("h_TSmaxCalib_HB", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HB = new TH2F("h_mapTSmaxCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HB = new TH2F("h_mapTSmaxCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HB = new TH1F("h_TSmeanCalib_HB", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HB = new TH2F("h_mapTSmeanCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HB = new TH2F("h_mapTSmeanCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HB = new TH1F("h_WidthCalib_HB", " ", 100, 0., 5.);
    h_mapWidthCalib047_HB = new TH2F("h_mapWidthCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HB = new TH2F("h_mapCapCalib047_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HB = new TH2F("h_mapWidthCalib_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HB = new TH2F("h_map_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HE = new TH1F("h_ADCCalib_HE", " ", 100, 10., 10000.);
    h_ADCCalib1_HE = new TH1F("h_ADCCalib1_HE", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HE = new TH2F("h_mapADCCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HE = new TH2F("h_mapADCCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HE = new TH1F("h_RatioCalib_HE", " ", 100, 0., 1.);
    h_mapRatioCalib047_HE = new TH2F("h_mapRatioCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HE = new TH2F("h_mapRatioCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HE = new TH1F("h_TSmaxCalib_HE", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HE = new TH2F("h_mapTSmaxCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HE = new TH2F("h_mapTSmaxCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HE = new TH1F("h_TSmeanCalib_HE", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HE = new TH2F("h_mapTSmeanCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HE = new TH2F("h_mapTSmeanCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HE = new TH1F("h_WidthCalib_HE", " ", 100, 0., 5.);
    h_mapWidthCalib047_HE = new TH2F("h_mapWidthCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HE = new TH2F("h_mapCapCalib047_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HE = new TH2F("h_mapWidthCalib_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HE = new TH2F("h_map_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HO = new TH1F("h_ADCCalib_HO", " ", 100, 10., 10000.);
    h_ADCCalib1_HO = new TH1F("h_ADCCalib1_HO", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HO = new TH2F("h_mapADCCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HO = new TH2F("h_mapADCCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HO = new TH1F("h_RatioCalib_HO", " ", 100, 0., 1.);
    h_mapRatioCalib047_HO = new TH2F("h_mapRatioCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HO = new TH2F("h_mapRatioCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HO = new TH1F("h_TSmaxCalib_HO", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HO = new TH2F("h_mapTSmaxCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HO = new TH2F("h_mapTSmaxCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HO = new TH1F("h_TSmeanCalib_HO", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HO = new TH2F("h_mapTSmeanCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HO = new TH2F("h_mapTSmeanCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HO = new TH1F("h_WidthCalib_HO", " ", 100, 0., 5.);
    h_mapWidthCalib047_HO = new TH2F("h_mapWidthCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HO = new TH2F("h_mapCapCalib047_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HO = new TH2F("h_mapWidthCalib_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HO = new TH2F("h_map_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_ADCCalib_HF = new TH1F("h_ADCCalib_HF", " ", 100, 10., 2000.);
    h_ADCCalib1_HF = new TH1F("h_ADCCalib1_HF", " ", 100, 0.1, 100.1);
    h_mapADCCalib047_HF = new TH2F("h_mapADCCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapADCCalib_HF = new TH2F("h_mapADCCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_RatioCalib_HF = new TH1F("h_RatioCalib_HF", " ", 100, 0., 1.);
    h_mapRatioCalib047_HF = new TH2F("h_mapRatioCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapRatioCalib_HF = new TH2F("h_mapRatioCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmaxCalib_HF = new TH1F("h_TSmaxCalib_HF", " ", 100, 0., 10.);
    h_mapTSmaxCalib047_HF = new TH2F("h_mapTSmaxCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmaxCalib_HF = new TH2F("h_mapTSmaxCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_TSmeanCalib_HF = new TH1F("h_TSmeanCalib_HF", " ", 100, 0., 10.);
    h_mapTSmeanCalib047_HF = new TH2F("h_mapTSmeanCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapTSmeanCalib_HF = new TH2F("h_mapTSmeanCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_WidthCalib_HF = new TH1F("h_WidthCalib_HF", " ", 100, 0., 5.);
    h_mapWidthCalib047_HF = new TH2F("h_mapWidthCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapCapCalib047_HF = new TH2F("h_mapCapCalib047_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapWidthCalib_HF = new TH2F("h_mapWidthCalib_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_map_HF = new TH2F("h_map_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_nls_per_run = new TH1F("h_nls_per_run", " ", 100, 0., 800.);
    h_nls_per_run10 = new TH1F("h_nls_per_run10", " ", 100, 0., 60.);
    h_nevents_per_LS = new TH1F("h_nevents_per_LS", " ", 100, 0., 600.);
    h_nevents_per_LSzoom = new TH1F("h_nevents_per_LSzoom", " ", 50, 0., 50.);
    h_nevents_per_eachLS = new TH1F("h_nevents_per_eachLS", " ", bac, 1., bac2);
    h_nevents_per_eachRealLS = new TH1F("h_nevents_per_eachRealLS", " ", bac, 1., bac2);
    h_lsnumber_per_eachLS = new TH1F("h_lsnumber_per_eachLS", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator0:
    float pst1 = 30.;
    h_sumPedestalLS1 = new TH1F("h_sumPedestalLS1", " ", 100, 0., pst1);
    h_2DsumPedestalLS1 = new TH2F("h_2DsumPedestalLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS1 = new TH1F("h_sumPedestalperLS1", " ", bac, 1., bac2);
    h_2D0sumPedestalLS1 = new TH2F("h_2D0sumPedestalLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS1 = new TH1F("h_sum0PedestalperLS1", " ", bac, 1., bac2);

    h_sumPedestalLS2 = new TH1F("h_sumPedestalLS2", " ", 100, 0., pst1);
    h_2DsumPedestalLS2 = new TH2F("h_2DsumPedestalLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS2 = new TH1F("h_sumPedestalperLS2", " ", bac, 1., bac2);
    h_2D0sumPedestalLS2 = new TH2F("h_2D0sumPedestalLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS2 = new TH1F("h_sum0PedestalperLS2", " ", bac, 1., bac2);

    h_sumPedestalLS3 = new TH1F("h_sumPedestalLS3", " ", 100, 0., pst1);
    h_2DsumPedestalLS3 = new TH2F("h_2DsumPedestalLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS3 = new TH1F("h_sumPedestalperLS3", " ", bac, 1., bac2);
    h_2D0sumPedestalLS3 = new TH2F("h_2D0sumPedestalLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS3 = new TH1F("h_sum0PedestalperLS3", " ", bac, 1., bac2);

    h_sumPedestalLS4 = new TH1F("h_sumPedestalLS4", " ", 100, 0., pst1);
    h_2DsumPedestalLS4 = new TH2F("h_2DsumPedestalLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS4 = new TH1F("h_sumPedestalperLS4", " ", bac, 1., bac2);
    h_2D0sumPedestalLS4 = new TH2F("h_2D0sumPedestalLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS4 = new TH1F("h_sum0PedestalperLS4", " ", bac, 1., bac2);

    h_sumPedestalLS5 = new TH1F("h_sumPedestalLS5", " ", 100, 0., pst1);
    h_2DsumPedestalLS5 = new TH2F("h_2DsumPedestalLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS5 = new TH1F("h_sumPedestalperLS5", " ", bac, 1., bac2);
    h_2D0sumPedestalLS5 = new TH2F("h_2D0sumPedestalLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS5 = new TH1F("h_sum0PedestalperLS5", " ", bac, 1., bac2);

    h_sumPedestalLS6 = new TH1F("h_sumPedestalLS6", " ", 100, 0., pst1);
    h_2DsumPedestalLS6 = new TH2F("h_2DsumPedestalLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS6 = new TH1F("h_sumPedestalperLS6", " ", bac, 1., bac2);
    h_2D0sumPedestalLS6 = new TH2F("h_2D0sumPedestalLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS6 = new TH1F("h_sum0PedestalperLS6", " ", bac, 1., bac2);

    h_sumPedestalLS7 = new TH1F("h_sumPedestalLS7", " ", 100, 0., pst1);
    h_2DsumPedestalLS7 = new TH2F("h_2DsumPedestalLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS7 = new TH1F("h_sumPedestalperLS7", " ", bac, 1., bac2);
    h_2D0sumPedestalLS7 = new TH2F("h_2D0sumPedestalLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS7 = new TH1F("h_sum0PedestalperLS7", " ", bac, 1., bac2);

    h_sumPedestalLS8 = new TH1F("h_sumPedestalLS8", " ", 100, 0., pst1);
    h_2DsumPedestalLS8 = new TH2F("h_2DsumPedestalLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumPedestalperLS8 = new TH1F("h_sumPedestalperLS8", " ", bac, 1., bac2);
    h_2D0sumPedestalLS8 = new TH2F("h_2D0sumPedestalLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0PedestalperLS8 = new TH1F("h_sum0PedestalperLS8", " ", bac, 1., bac2);

    //--------------------------------------------------
    // for estimator1:
    h_sumADCAmplLS1copy1 = new TH1F("h_sumADCAmplLS1copy1", " ", 100, 0., 10000);
    h_sumADCAmplLS1copy2 = new TH1F("h_sumADCAmplLS1copy2", " ", 100, 0., 20000);
    h_sumADCAmplLS1copy3 = new TH1F("h_sumADCAmplLS1copy3", " ", 100, 0., 50000);
    h_sumADCAmplLS1copy4 = new TH1F("h_sumADCAmplLS1copy4", " ", 100, 0., 100000);
    h_sumADCAmplLS1copy5 = new TH1F("h_sumADCAmplLS1copy5", " ", 100, 0., 150000);
    h_sumADCAmplLS1 = new TH1F("h_sumADCAmplLS1", " ", 100, 0., lsdep_estimator1_HBdepth1_);
    h_2DsumADCAmplLS1 = new TH2F("h_2DsumADCAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS1_LSselected = new TH2F("h_2DsumADCAmplLS1_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS1 = new TH1F("h_sumADCAmplperLS1", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS1 = new TH1F("h_sumCutADCAmplperLS1", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS1 = new TH2F("h_2D0sumADCAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS1 = new TH1F("h_sum0ADCAmplperLS1", " ", bac, 1., bac2);

    h_sumADCAmplLS2 = new TH1F("h_sumADCAmplLS2", " ", 100, 0., lsdep_estimator1_HBdepth2_);
    h_2DsumADCAmplLS2 = new TH2F("h_2DsumADCAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS2_LSselected = new TH2F("h_2DsumADCAmplLS2_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS2 = new TH1F("h_sumADCAmplperLS2", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS2 = new TH1F("h_sumCutADCAmplperLS2", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS2 = new TH2F("h_2D0sumADCAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS2 = new TH1F("h_sum0ADCAmplperLS2", " ", bac, 1., bac2);

    h_sumADCAmplLS3 = new TH1F("h_sumADCAmplLS3", " ", 100, 0., lsdep_estimator1_HEdepth1_);
    h_2DsumADCAmplLS3 = new TH2F("h_2DsumADCAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS3_LSselected = new TH2F("h_2DsumADCAmplLS3_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS3 = new TH1F("h_sumADCAmplperLS3", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS3 = new TH1F("h_sumCutADCAmplperLS3", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS3 = new TH2F("h_2D0sumADCAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS3 = new TH1F("h_sum0ADCAmplperLS3", " ", bac, 1., bac2);

    h_sumADCAmplLS4 = new TH1F("h_sumADCAmplLS4", " ", 100, 0., lsdep_estimator1_HEdepth2_);
    h_2DsumADCAmplLS4 = new TH2F("h_2DsumADCAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS4_LSselected = new TH2F("h_2DsumADCAmplLS4_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS4 = new TH1F("h_sumADCAmplperLS4", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS4 = new TH1F("h_sumCutADCAmplperLS4", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS4 = new TH2F("h_2D0sumADCAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS4 = new TH1F("h_sum0ADCAmplperLS4", " ", bac, 1., bac2);

    h_sumADCAmplLS5 = new TH1F("h_sumADCAmplLS5", " ", 100, 0., lsdep_estimator1_HEdepth3_);
    h_2DsumADCAmplLS5 = new TH2F("h_2DsumADCAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS5_LSselected = new TH2F("h_2DsumADCAmplLS5_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS5 = new TH1F("h_sumADCAmplperLS5", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS5 = new TH1F("h_sumCutADCAmplperLS5", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS5 = new TH2F("h_2D0sumADCAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS5 = new TH1F("h_sum0ADCAmplperLS5", " ", bac, 1., bac2);
    // HE upgrade depth4
    h_sumADCAmplperLSdepth4HEu = new TH1F("h_sumADCAmplperLSdepth4HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth4HEu = new TH1F("h_sumCutADCAmplperLSdepth4HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth4HEu = new TH1F("h_sum0ADCAmplperLSdepth4HEu", " ", bac, 1., bac2);

    // HE upgrade depth5
    h_sumADCAmplperLSdepth5HEu = new TH1F("h_sumADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth5HEu = new TH1F("h_sumCutADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth5HEu = new TH1F("h_sum0ADCAmplperLSdepth5HEu", " ", bac, 1., bac2);
    // HE upgrade depth6
    h_sumADCAmplperLSdepth6HEu = new TH1F("h_sumADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth6HEu = new TH1F("h_sumCutADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth6HEu = new TH1F("h_sum0ADCAmplperLSdepth6HEu", " ", bac, 1., bac2);
    // HE upgrade depth7
    h_sumADCAmplperLSdepth7HEu = new TH1F("h_sumADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth7HEu = new TH1F("h_sumCutADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth7HEu = new TH1F("h_sum0ADCAmplperLSdepth7HEu", " ", bac, 1., bac2);
    // for HE gain stability vs LS:
    h_2DsumADCAmplLSdepth4HEu = new TH2F("h_2DsumADCAmplLSdepth4HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HEu = new TH2F("h_2D0sumADCAmplLSdepth4HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth5HEu = new TH2F("h_2DsumADCAmplLSdepth5HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth5HEu = new TH2F("h_2D0sumADCAmplLSdepth5HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth6HEu = new TH2F("h_2DsumADCAmplLSdepth6HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth6HEu = new TH2F("h_2D0sumADCAmplLSdepth6HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth7HEu = new TH2F("h_2DsumADCAmplLSdepth7HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth7HEu = new TH2F("h_2D0sumADCAmplLSdepth7HEu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth3HFu = new TH2F("h_2DsumADCAmplLSdepth3HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth3HFu = new TH2F("h_2D0sumADCAmplLSdepth3HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth4HFu = new TH2F("h_2DsumADCAmplLSdepth4HFu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HFu = new TH2F("h_2D0sumADCAmplLSdepth4HFu", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumADCAmplLS6 = new TH1F("h_sumADCAmplLS6", " ", 100, 0., lsdep_estimator1_HFdepth1_);
    h_2DsumADCAmplLS6 = new TH2F("h_2DsumADCAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS6_LSselected = new TH2F("h_2DsumADCAmplLS6_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLS6 = new TH2F("h_2D0sumADCAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS6 = new TH1F("h_sumADCAmplperLS6", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS6 = new TH1F("h_sumCutADCAmplperLS6", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6 = new TH1F("h_sum0ADCAmplperLS6", " ", bac, 1., bac2);
    // HF upgrade depth3
    h_sumADCAmplperLS6u = new TH1F("h_sumADCAmplperLS6u", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS6u = new TH1F("h_sumCutADCAmplperLS6u", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6u = new TH1F("h_sum0ADCAmplperLS6u", " ", bac, 1., bac2);

    h_sumADCAmplLS7 = new TH1F("h_sumADCAmplLS7", " ", 100, 0., lsdep_estimator1_HFdepth2_);
    h_2DsumADCAmplLS7 = new TH2F("h_2DsumADCAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS7_LSselected = new TH2F("h_2DsumADCAmplLS7_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLS7 = new TH2F("h_2D0sumADCAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS7 = new TH1F("h_sumADCAmplperLS7", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS7 = new TH1F("h_sumCutADCAmplperLS7", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS7 = new TH1F("h_sum0ADCAmplperLS7", " ", bac, 1., bac2);
    // HF upgrade depth4
    h_sumADCAmplperLS7u = new TH1F("h_sumADCAmplperLS7u", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS7u = new TH1F("h_sumCutADCAmplperLS7u", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS7u = new TH1F("h_sum0ADCAmplperLS7u", " ", bac, 1., bac2);

    h_sumADCAmplLS8 = new TH1F("h_sumADCAmplLS8", " ", 100, 0., lsdep_estimator1_HOdepth4_);
    h_2DsumADCAmplLS8 = new TH2F("h_2DsumADCAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLS8_LSselected = new TH2F("h_2DsumADCAmplLS8_LSselected", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumADCAmplperLS8 = new TH1F("h_sumADCAmplperLS8", " ", bac, 1., bac2);
    h_sumCutADCAmplperLS8 = new TH1F("h_sumCutADCAmplperLS8", " ", bac, 1., bac2);
    h_2D0sumADCAmplLS8 = new TH2F("h_2D0sumADCAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0ADCAmplperLS8 = new TH1F("h_sum0ADCAmplperLS8", " ", bac, 1., bac2);

    // HB upgrade depth3
    h_sumADCAmplperLSdepth3HBu = new TH1F("h_sumADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth3HBu = new TH1F("h_sumCutADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth3HBu = new TH1F("h_sum0ADCAmplperLSdepth3HBu", " ", bac, 1., bac2);
    // HB upgrade depth4
    h_sumADCAmplperLSdepth4HBu = new TH1F("h_sumADCAmplperLSdepth4HBu", " ", bac, 1., bac2);
    h_sumCutADCAmplperLSdepth4HBu = new TH1F("h_sumCutADCAmplperLSdepth4HBu", " ", bac, 1., bac2);
    h_sum0ADCAmplperLSdepth4HBu = new TH1F("h_sum0ADCAmplperLSdepth4HBu", " ", bac, 1., bac2);

    // for HB gain stability vs LS:
    h_2DsumADCAmplLSdepth3HBu = new TH2F("h_2DsumADCAmplLSdepth3HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth3HBu = new TH2F("h_2D0sumADCAmplLSdepth3HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumADCAmplLSdepth4HBu = new TH2F("h_2DsumADCAmplLSdepth4HBu", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0sumADCAmplLSdepth4HBu = new TH2F("h_2D0sumADCAmplLSdepth4HBu", " ", neta, -41., 41., nphi, 0., bphi);

    // error-A for HB( depth1 only)
    h_sumADCAmplperLS1_P1 = new TH1F("h_sumADCAmplperLS1_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_P1 = new TH1F("h_sum0ADCAmplperLS1_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_P2 = new TH1F("h_sumADCAmplperLS1_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_P2 = new TH1F("h_sum0ADCAmplperLS1_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_M1 = new TH1F("h_sumADCAmplperLS1_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_M1 = new TH1F("h_sum0ADCAmplperLS1_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS1_M2 = new TH1F("h_sumADCAmplperLS1_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS1_M2 = new TH1F("h_sum0ADCAmplperLS1_M2", " ", bac, 1., bac2);

    // error-A for HE( depth1 only)
    h_sumADCAmplperLS3_P1 = new TH1F("h_sumADCAmplperLS3_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_P1 = new TH1F("h_sum0ADCAmplperLS3_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_P2 = new TH1F("h_sumADCAmplperLS3_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_P2 = new TH1F("h_sum0ADCAmplperLS3_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_M1 = new TH1F("h_sumADCAmplperLS3_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_M1 = new TH1F("h_sum0ADCAmplperLS3_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS3_M2 = new TH1F("h_sumADCAmplperLS3_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS3_M2 = new TH1F("h_sum0ADCAmplperLS3_M2", " ", bac, 1., bac2);

    // error-A for HF( depth1 only)
    h_sumADCAmplperLS6_P1 = new TH1F("h_sumADCAmplperLS6_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_P1 = new TH1F("h_sum0ADCAmplperLS6_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_P2 = new TH1F("h_sumADCAmplperLS6_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_P2 = new TH1F("h_sum0ADCAmplperLS6_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_M1 = new TH1F("h_sumADCAmplperLS6_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_M1 = new TH1F("h_sum0ADCAmplperLS6_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS6_M2 = new TH1F("h_sumADCAmplperLS6_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS6_M2 = new TH1F("h_sum0ADCAmplperLS6_M2", " ", bac, 1., bac2);

    // error-A for HO( depth4 only)
    h_sumADCAmplperLS8_P1 = new TH1F("h_sumADCAmplperLS8_P1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_P1 = new TH1F("h_sum0ADCAmplperLS8_P1", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_P2 = new TH1F("h_sumADCAmplperLS8_P2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_P2 = new TH1F("h_sum0ADCAmplperLS8_P2", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_M1 = new TH1F("h_sumADCAmplperLS8_M1", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_M1 = new TH1F("h_sum0ADCAmplperLS8_M1", " ", bac, 1., bac2);
    h_sumADCAmplperLS8_M2 = new TH1F("h_sumADCAmplperLS8_M2", " ", bac, 1., bac2);
    h_sum0ADCAmplperLS8_M2 = new TH1F("h_sum0ADCAmplperLS8_M2", " ", bac, 1., bac2);

    //--------------------------------------------------
    h_sumTSmeanALS1 = new TH1F("h_sumTSmeanALS1", " ", 100, 0., lsdep_estimator2_HBdepth1_);
    h_2DsumTSmeanALS1 = new TH2F("h_2DsumTSmeanALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS1 = new TH1F("h_sumTSmeanAperLS1", " ", bac, 1., bac2);
    h_sumTSmeanAperLS1_LSselected = new TH1F("h_sumTSmeanAperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS1 = new TH1F("h_sumCutTSmeanAperLS1", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS1 = new TH2F("h_2D0sumTSmeanALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS1 = new TH1F("h_sum0TSmeanAperLS1", " ", bac, 1., bac2);

    h_sumTSmeanALS2 = new TH1F("h_sumTSmeanALS2", " ", 100, 0., lsdep_estimator2_HBdepth2_);
    h_2DsumTSmeanALS2 = new TH2F("h_2DsumTSmeanALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS2 = new TH1F("h_sumTSmeanAperLS2", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS2 = new TH1F("h_sumCutTSmeanAperLS2", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS2 = new TH2F("h_2D0sumTSmeanALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS2 = new TH1F("h_sum0TSmeanAperLS2", " ", bac, 1., bac2);

    h_sumTSmeanALS3 = new TH1F("h_sumTSmeanALS3", " ", 100, 0., lsdep_estimator2_HEdepth1_);
    h_2DsumTSmeanALS3 = new TH2F("h_2DsumTSmeanALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS3 = new TH1F("h_sumTSmeanAperLS3", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS3 = new TH1F("h_sumCutTSmeanAperLS3", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS3 = new TH2F("h_2D0sumTSmeanALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS3 = new TH1F("h_sum0TSmeanAperLS3", " ", bac, 1., bac2);

    h_sumTSmeanALS4 = new TH1F("h_sumTSmeanALS4", " ", 100, 0., lsdep_estimator2_HEdepth2_);
    h_2DsumTSmeanALS4 = new TH2F("h_2DsumTSmeanALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS4 = new TH1F("h_sumTSmeanAperLS4", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS4 = new TH1F("h_sumCutTSmeanAperLS4", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS4 = new TH2F("h_2D0sumTSmeanALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS4 = new TH1F("h_sum0TSmeanAperLS4", " ", bac, 1., bac2);

    h_sumTSmeanALS5 = new TH1F("h_sumTSmeanALS5", " ", 100, 0., lsdep_estimator2_HEdepth3_);
    h_2DsumTSmeanALS5 = new TH2F("h_2DsumTSmeanALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS5 = new TH1F("h_sumTSmeanAperLS5", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS5 = new TH1F("h_sumCutTSmeanAperLS5", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS5 = new TH2F("h_2D0sumTSmeanALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS5 = new TH1F("h_sum0TSmeanAperLS5", " ", bac, 1., bac2);

    h_sumTSmeanALS6 = new TH1F("h_sumTSmeanALS6", " ", 100, 0., lsdep_estimator2_HFdepth1_);
    h_2DsumTSmeanALS6 = new TH2F("h_2DsumTSmeanALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS6 = new TH1F("h_sumTSmeanAperLS6", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS6 = new TH1F("h_sumCutTSmeanAperLS6", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS6 = new TH2F("h_2D0sumTSmeanALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS6 = new TH1F("h_sum0TSmeanAperLS6", " ", bac, 1., bac2);

    h_sumTSmeanALS7 = new TH1F("h_sumTSmeanALS7", " ", 100, 0., lsdep_estimator2_HFdepth2_);
    h_2DsumTSmeanALS7 = new TH2F("h_2DsumTSmeanALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS7 = new TH1F("h_sumTSmeanAperLS7", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS7 = new TH1F("h_sumCutTSmeanAperLS7", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS7 = new TH2F("h_2D0sumTSmeanALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS7 = new TH1F("h_sum0TSmeanAperLS7", " ", bac, 1., bac2);

    h_sumTSmeanALS8 = new TH1F("h_sumTSmeanALS8", " ", 100, 0., lsdep_estimator2_HOdepth4_);
    h_2DsumTSmeanALS8 = new TH2F("h_2DsumTSmeanALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmeanAperLS8 = new TH1F("h_sumTSmeanAperLS8", " ", bac, 1., bac2);
    h_sumCutTSmeanAperLS8 = new TH1F("h_sumCutTSmeanAperLS8", " ", bac, 1., bac2);
    h_2D0sumTSmeanALS8 = new TH2F("h_2D0sumTSmeanALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmeanAperLS8 = new TH1F("h_sum0TSmeanAperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator3:
    //  float est3 = 10.0;
    h_sumTSmaxALS1 = new TH1F("h_sumTSmaxALS1", " ", 100, 0., lsdep_estimator3_HBdepth1_);
    h_2DsumTSmaxALS1 = new TH2F("h_2DsumTSmaxALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS1 = new TH1F("h_sumTSmaxAperLS1", " ", bac, 1., bac2);
    h_sumTSmaxAperLS1_LSselected = new TH1F("h_sumTSmaxAperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS1 = new TH1F("h_sumCutTSmaxAperLS1", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS1 = new TH2F("h_2D0sumTSmaxALS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS1 = new TH1F("h_sum0TSmaxAperLS1", " ", bac, 1., bac2);

    h_sumTSmaxALS2 = new TH1F("h_sumTSmaxALS2", " ", 100, 0., lsdep_estimator3_HBdepth2_);
    h_2DsumTSmaxALS2 = new TH2F("h_2DsumTSmaxALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS2 = new TH1F("h_sumTSmaxAperLS2", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS2 = new TH1F("h_sumCutTSmaxAperLS2", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS2 = new TH2F("h_2D0sumTSmaxALS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS2 = new TH1F("h_sum0TSmaxAperLS2", " ", bac, 1., bac2);

    h_sumTSmaxALS3 = new TH1F("h_sumTSmaxALS3", " ", 100, 0., lsdep_estimator3_HEdepth1_);
    h_2DsumTSmaxALS3 = new TH2F("h_2DsumTSmaxALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS3 = new TH1F("h_sumTSmaxAperLS3", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS3 = new TH1F("h_sumCutTSmaxAperLS3", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS3 = new TH2F("h_2D0sumTSmaxALS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS3 = new TH1F("h_sum0TSmaxAperLS3", " ", bac, 1., bac2);

    h_sumTSmaxALS4 = new TH1F("h_sumTSmaxALS4", " ", 100, 0., lsdep_estimator3_HEdepth2_);
    h_2DsumTSmaxALS4 = new TH2F("h_2DsumTSmaxALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS4 = new TH1F("h_sumTSmaxAperLS4", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS4 = new TH1F("h_sumCutTSmaxAperLS4", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS4 = new TH2F("h_2D0sumTSmaxALS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS4 = new TH1F("h_sum0TSmaxAperLS4", " ", bac, 1., bac2);

    h_sumTSmaxALS5 = new TH1F("h_sumTSmaxALS5", " ", 100, 0., lsdep_estimator3_HEdepth3_);
    h_2DsumTSmaxALS5 = new TH2F("h_2DsumTSmaxALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS5 = new TH1F("h_sumTSmaxAperLS5", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS5 = new TH1F("h_sumCutTSmaxAperLS5", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS5 = new TH2F("h_2D0sumTSmaxALS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS5 = new TH1F("h_sum0TSmaxAperLS5", " ", bac, 1., bac2);

    h_sumTSmaxALS6 = new TH1F("h_sumTSmaxALS6", " ", 100, 0., lsdep_estimator3_HFdepth1_);
    h_2DsumTSmaxALS6 = new TH2F("h_2DsumTSmaxALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS6 = new TH1F("h_sumTSmaxAperLS6", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS6 = new TH1F("h_sumCutTSmaxAperLS6", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS6 = new TH2F("h_2D0sumTSmaxALS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS6 = new TH1F("h_sum0TSmaxAperLS6", " ", bac, 1., bac2);

    h_sumTSmaxALS7 = new TH1F("h_sumTSmaxALS7", " ", 100, 0., lsdep_estimator3_HFdepth2_);
    h_2DsumTSmaxALS7 = new TH2F("h_2DsumTSmaxALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS7 = new TH1F("h_sumTSmaxAperLS7", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS7 = new TH1F("h_sumCutTSmaxAperLS7", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS7 = new TH2F("h_2D0sumTSmaxALS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS7 = new TH1F("h_sum0TSmaxAperLS7", " ", bac, 1., bac2);

    h_sumTSmaxALS8 = new TH1F("h_sumTSmaxALS8", " ", 100, 0., lsdep_estimator3_HOdepth4_);
    h_2DsumTSmaxALS8 = new TH2F("h_2DsumTSmaxALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumTSmaxAperLS8 = new TH1F("h_sumTSmaxAperLS8", " ", bac, 1., bac2);
    h_sumCutTSmaxAperLS8 = new TH1F("h_sumCutTSmaxAperLS8", " ", bac, 1., bac2);
    h_2D0sumTSmaxALS8 = new TH2F("h_2D0sumTSmaxALS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0TSmaxAperLS8 = new TH1F("h_sum0TSmaxAperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator4:
    //  float est4 = 3.4;
    //  float est41= 2.0;
    h_sumAmplitudeLS1 = new TH1F("h_sumAmplitudeLS1", " ", 100, 0.0, lsdep_estimator4_HBdepth1_);
    h_2DsumAmplitudeLS1 = new TH2F("h_2DsumAmplitudeLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS1 = new TH1F("h_sumAmplitudeperLS1", " ", bac, 1., bac2);
    h_sumAmplitudeperLS1_LSselected = new TH1F("h_sumAmplitudeperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS1 = new TH1F("h_sumCutAmplitudeperLS1", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS1 = new TH2F("h_2D0sumAmplitudeLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS1 = new TH1F("h_sum0AmplitudeperLS1", " ", bac, 1., bac2);

    h_sumAmplitudeLS2 = new TH1F("h_sumAmplitudeLS2", " ", 100, 0.0, lsdep_estimator4_HBdepth2_);
    h_2DsumAmplitudeLS2 = new TH2F("h_2DsumAmplitudeLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS2 = new TH1F("h_sumAmplitudeperLS2", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS2 = new TH1F("h_sumCutAmplitudeperLS2", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS2 = new TH2F("h_2D0sumAmplitudeLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS2 = new TH1F("h_sum0AmplitudeperLS2", " ", bac, 1., bac2);

    h_sumAmplitudeLS3 = new TH1F("h_sumAmplitudeLS3", " ", 100, 0.0, lsdep_estimator4_HEdepth1_);
    h_2DsumAmplitudeLS3 = new TH2F("h_2DsumAmplitudeLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS3 = new TH1F("h_sumAmplitudeperLS3", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS3 = new TH1F("h_sumCutAmplitudeperLS3", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS3 = new TH2F("h_2D0sumAmplitudeLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS3 = new TH1F("h_sum0AmplitudeperLS3", " ", bac, 1., bac2);

    h_sumAmplitudeLS4 = new TH1F("h_sumAmplitudeLS4", " ", 100, 0.0, lsdep_estimator4_HEdepth2_);
    h_2DsumAmplitudeLS4 = new TH2F("h_2DsumAmplitudeLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS4 = new TH1F("h_sumAmplitudeperLS4", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS4 = new TH1F("h_sumCutAmplitudeperLS4", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS4 = new TH2F("h_2D0sumAmplitudeLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS4 = new TH1F("h_sum0AmplitudeperLS4", " ", bac, 1., bac2);

    h_sumAmplitudeLS5 = new TH1F("h_sumAmplitudeLS5", " ", 100, 0.0, lsdep_estimator4_HEdepth3_);
    h_2DsumAmplitudeLS5 = new TH2F("h_2DsumAmplitudeLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS5 = new TH1F("h_sumAmplitudeperLS5", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS5 = new TH1F("h_sumCutAmplitudeperLS5", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS5 = new TH2F("h_2D0sumAmplitudeLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS5 = new TH1F("h_sum0AmplitudeperLS5", " ", bac, 1., bac2);

    h_sumAmplitudeLS6 = new TH1F("h_sumAmplitudeLS6", " ", 100, 0., lsdep_estimator4_HFdepth1_);
    h_2DsumAmplitudeLS6 = new TH2F("h_2DsumAmplitudeLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS6 = new TH1F("h_sumAmplitudeperLS6", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS6 = new TH1F("h_sumCutAmplitudeperLS6", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS6 = new TH2F("h_2D0sumAmplitudeLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS6 = new TH1F("h_sum0AmplitudeperLS6", " ", bac, 1., bac2);

    h_sumAmplitudeLS7 = new TH1F("h_sumAmplitudeLS7", " ", 100, 0., lsdep_estimator4_HFdepth2_);
    h_2DsumAmplitudeLS7 = new TH2F("h_2DsumAmplitudeLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS7 = new TH1F("h_sumAmplitudeperLS7", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS7 = new TH1F("h_sumCutAmplitudeperLS7", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS7 = new TH2F("h_2D0sumAmplitudeLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS7 = new TH1F("h_sum0AmplitudeperLS7", " ", bac, 1., bac2);

    h_sumAmplitudeLS8 = new TH1F("h_sumAmplitudeLS8", " ", 100, 0., lsdep_estimator4_HOdepth4_);
    h_2DsumAmplitudeLS8 = new TH2F("h_2DsumAmplitudeLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplitudeperLS8 = new TH1F("h_sumAmplitudeperLS8", " ", bac, 1., bac2);
    h_sumCutAmplitudeperLS8 = new TH1F("h_sumCutAmplitudeperLS8", " ", bac, 1., bac2);
    h_2D0sumAmplitudeLS8 = new TH2F("h_2D0sumAmplitudeLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplitudeperLS8 = new TH1F("h_sum0AmplitudeperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator5:
    //  float est5 = 0.6;
    //  float est51= 1.0;
    //  float est52= 0.8;
    h_sumAmplLS1 = new TH1F("h_sumAmplLS1", " ", 100, 0.0, lsdep_estimator5_HBdepth1_);
    h_2DsumAmplLS1 = new TH2F("h_2DsumAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS1 = new TH1F("h_sumAmplperLS1", " ", bac, 1., bac2);
    h_sumAmplperLS1_LSselected = new TH1F("h_sumAmplperLS1_LSselected", " ", bac, 1., bac2);
    h_sumCutAmplperLS1 = new TH1F("h_sumCutAmplperLS1", " ", bac, 1., bac2);
    h_2D0sumAmplLS1 = new TH2F("h_2D0sumAmplLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS1 = new TH1F("h_sum0AmplperLS1", " ", bac, 1., bac2);

    h_sumAmplLS2 = new TH1F("h_sumAmplLS2", " ", 100, 0.0, lsdep_estimator5_HBdepth2_);
    h_2DsumAmplLS2 = new TH2F("h_2DsumAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS2 = new TH1F("h_sumAmplperLS2", " ", bac, 1., bac2);
    h_sumCutAmplperLS2 = new TH1F("h_sumCutAmplperLS2", " ", bac, 1., bac2);
    h_2D0sumAmplLS2 = new TH2F("h_2D0sumAmplLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS2 = new TH1F("h_sum0AmplperLS2", " ", bac, 1., bac2);

    h_sumAmplLS3 = new TH1F("h_sumAmplLS3", " ", 100, 0.0, lsdep_estimator5_HEdepth1_);
    h_2DsumAmplLS3 = new TH2F("h_2DsumAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS3 = new TH1F("h_sumAmplperLS3", " ", bac, 1., bac2);
    h_sumCutAmplperLS3 = new TH1F("h_sumCutAmplperLS3", " ", bac, 1., bac2);
    h_2D0sumAmplLS3 = new TH2F("h_2D0sumAmplLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS3 = new TH1F("h_sum0AmplperLS3", " ", bac, 1., bac2);

    h_sumAmplLS4 = new TH1F("h_sumAmplLS4", " ", 100, 0.0, lsdep_estimator5_HEdepth2_);
    h_2DsumAmplLS4 = new TH2F("h_2DsumAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS4 = new TH1F("h_sumAmplperLS4", " ", bac, 1., bac2);
    h_sumCutAmplperLS4 = new TH1F("h_sumCutAmplperLS4", " ", bac, 1., bac2);
    h_2D0sumAmplLS4 = new TH2F("h_2D0sumAmplLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS4 = new TH1F("h_sum0AmplperLS4", " ", bac, 1., bac2);

    h_sumAmplLS5 = new TH1F("h_sumAmplLS5", " ", 100, 0.0, lsdep_estimator5_HEdepth3_);
    h_2DsumAmplLS5 = new TH2F("h_2DsumAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS5 = new TH1F("h_sumAmplperLS5", " ", bac, 1., bac2);
    h_sumCutAmplperLS5 = new TH1F("h_sumCutAmplperLS5", " ", bac, 1., bac2);
    h_2D0sumAmplLS5 = new TH2F("h_2D0sumAmplLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS5 = new TH1F("h_sum0AmplperLS5", " ", bac, 1., bac2);

    h_sumAmplLS6 = new TH1F("h_sumAmplLS6", " ", 100, 0.0, lsdep_estimator5_HFdepth1_);
    h_2DsumAmplLS6 = new TH2F("h_2DsumAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS6 = new TH1F("h_sumAmplperLS6", " ", bac, 1., bac2);
    h_sumCutAmplperLS6 = new TH1F("h_sumCutAmplperLS6", " ", bac, 1., bac2);
    h_2D0sumAmplLS6 = new TH2F("h_2D0sumAmplLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS6 = new TH1F("h_sum0AmplperLS6", " ", bac, 1., bac2);

    h_RatioOccupancy_HBP = new TH1F("h_RatioOccupancy_HBP", " ", bac, 1., bac2);
    h_RatioOccupancy_HBM = new TH1F("h_RatioOccupancy_HBM", " ", bac, 1., bac2);
    h_RatioOccupancy_HEP = new TH1F("h_RatioOccupancy_HEP", " ", bac, 1., bac2);
    h_RatioOccupancy_HEM = new TH1F("h_RatioOccupancy_HEM", " ", bac, 1., bac2);
    h_RatioOccupancy_HOP = new TH1F("h_RatioOccupancy_HOP", " ", bac, 1., bac2);
    h_RatioOccupancy_HOM = new TH1F("h_RatioOccupancy_HOM", " ", bac, 1., bac2);
    h_RatioOccupancy_HFP = new TH1F("h_RatioOccupancy_HFP", " ", bac, 1., bac2);
    h_RatioOccupancy_HFM = new TH1F("h_RatioOccupancy_HFM", " ", bac, 1., bac2);

    h_sumAmplLS7 = new TH1F("h_sumAmplLS7", " ", 100, 0.0, lsdep_estimator5_HFdepth2_);
    h_2DsumAmplLS7 = new TH2F("h_2DsumAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS7 = new TH1F("h_sumAmplperLS7", " ", bac, 1., bac2);
    h_sumCutAmplperLS7 = new TH1F("h_sumCutAmplperLS7", " ", bac, 1., bac2);
    h_2D0sumAmplLS7 = new TH2F("h_2D0sumAmplLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS7 = new TH1F("h_sum0AmplperLS7", " ", bac, 1., bac2);

    h_sumAmplLS8 = new TH1F("h_sumAmplLS8", " ", 100, 0.0, lsdep_estimator5_HOdepth4_);
    h_2DsumAmplLS8 = new TH2F("h_2DsumAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumAmplperLS8 = new TH1F("h_sumAmplperLS8", " ", bac, 1., bac2);
    h_sumCutAmplperLS8 = new TH1F("h_sumCutAmplperLS8", " ", bac, 1., bac2);
    h_2D0sumAmplLS8 = new TH2F("h_2D0sumAmplLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_sum0AmplperLS8 = new TH1F("h_sum0AmplperLS8", " ", bac, 1., bac2);
    //--------------------------------------------------
    // for estimator6:
    h_sumErrorBLS1 = new TH1F("h_sumErrorBLS1", " ", 10, 0., 10.);
    h_sumErrorBperLS1 = new TH1F("h_sumErrorBperLS1", " ", bac, 1., bac2);
    h_sum0ErrorBperLS1 = new TH1F("h_sum0ErrorBperLS1", " ", bac, 1., bac2);
    h_2D0sumErrorBLS1 = new TH2F("h_2D0sumErrorBLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS1 = new TH2F("h_2DsumErrorBLS1", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS2 = new TH1F("h_sumErrorBLS2", " ", 10, 0., 10.);
    h_sumErrorBperLS2 = new TH1F("h_sumErrorBperLS2", " ", bac, 1., bac2);
    h_sum0ErrorBperLS2 = new TH1F("h_sum0ErrorBperLS2", " ", bac, 1., bac2);
    h_2D0sumErrorBLS2 = new TH2F("h_2D0sumErrorBLS2", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS2 = new TH2F("h_2DsumErrorBLS2", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS3 = new TH1F("h_sumErrorBLS3", " ", 10, 0., 10.);
    h_sumErrorBperLS3 = new TH1F("h_sumErrorBperLS3", " ", bac, 1., bac2);
    h_sum0ErrorBperLS3 = new TH1F("h_sum0ErrorBperLS3", " ", bac, 1., bac2);
    h_2D0sumErrorBLS3 = new TH2F("h_2D0sumErrorBLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS3 = new TH2F("h_2DsumErrorBLS3", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS4 = new TH1F("h_sumErrorBLS4", " ", 10, 0., 10.);
    h_sumErrorBperLS4 = new TH1F("h_sumErrorBperLS4", " ", bac, 1., bac2);
    h_sum0ErrorBperLS4 = new TH1F("h_sum0ErrorBperLS4", " ", bac, 1., bac2);
    h_2D0sumErrorBLS4 = new TH2F("h_2D0sumErrorBLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS4 = new TH2F("h_2DsumErrorBLS4", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS5 = new TH1F("h_sumErrorBLS5", " ", 10, 0., 10.);
    h_sumErrorBperLS5 = new TH1F("h_sumErrorBperLS5", " ", bac, 1., bac2);
    h_sum0ErrorBperLS5 = new TH1F("h_sum0ErrorBperLS5", " ", bac, 1., bac2);
    h_2D0sumErrorBLS5 = new TH2F("h_2D0sumErrorBLS5", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS5 = new TH2F("h_2DsumErrorBLS5", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS6 = new TH1F("h_sumErrorBLS6", " ", 10, 0., 10.);
    h_sumErrorBperLS6 = new TH1F("h_sumErrorBperLS6", " ", bac, 1., bac2);
    h_sum0ErrorBperLS6 = new TH1F("h_sum0ErrorBperLS6", " ", bac, 1., bac2);
    h_2D0sumErrorBLS6 = new TH2F("h_2D0sumErrorBLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS6 = new TH2F("h_2DsumErrorBLS6", " ", neta, -41., 41., nphi, 0., bphi);
    h_sumErrorBLS7 = new TH1F("h_sumErrorBLS7", " ", 10, 0., 10.);
    h_sumErrorBperLS7 = new TH1F("h_sumErrorBperLS7", " ", bac, 1., bac2);
    h_sum0ErrorBperLS7 = new TH1F("h_sum0ErrorBperLS7", " ", bac, 1., bac2);
    h_2D0sumErrorBLS7 = new TH2F("h_2D0sumErrorBLS7", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS7 = new TH2F("h_2DsumErrorBLS7", " ", neta, -41., 41., nphi, 0., bphi);

    h_sumErrorBLS8 = new TH1F("h_sumErrorBLS8", " ", 10, 0., 10.);
    h_sumErrorBperLS8 = new TH1F("h_sumErrorBperLS8", " ", bac, 1., bac2);
    h_sum0ErrorBperLS8 = new TH1F("h_sum0ErrorBperLS8", " ", bac, 1., bac2);
    h_2D0sumErrorBLS8 = new TH2F("h_2D0sumErrorBLS8", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DsumErrorBLS8 = new TH2F("h_2DsumErrorBLS8", " ", neta, -41., 41., nphi, 0., bphi);

    //--------------------------------------------------
    // for averSIGNALOCCUPANCY :
    h_averSIGNALoccupancy_HB = new TH1F("h_averSIGNALoccupancy_HB", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HE = new TH1F("h_averSIGNALoccupancy_HE", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HF = new TH1F("h_averSIGNALoccupancy_HF", " ", bac, 1., bac2);
    h_averSIGNALoccupancy_HO = new TH1F("h_averSIGNALoccupancy_HO", " ", bac, 1., bac2);

    // for averSIGNALsumamplitude :
    h_averSIGNALsumamplitude_HB = new TH1F("h_averSIGNALsumamplitude_HB", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HE = new TH1F("h_averSIGNALsumamplitude_HE", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HF = new TH1F("h_averSIGNALsumamplitude_HF", " ", bac, 1., bac2);
    h_averSIGNALsumamplitude_HO = new TH1F("h_averSIGNALsumamplitude_HO", " ", bac, 1., bac2);

    // for averNOSIGNALOCCUPANCY :
    h_averNOSIGNALoccupancy_HB = new TH1F("h_averNOSIGNALoccupancy_HB", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HE = new TH1F("h_averNOSIGNALoccupancy_HE", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HF = new TH1F("h_averNOSIGNALoccupancy_HF", " ", bac, 1., bac2);
    h_averNOSIGNALoccupancy_HO = new TH1F("h_averNOSIGNALoccupancy_HO", " ", bac, 1., bac2);

    // for averNOSIGNALsumamplitude :
    h_averNOSIGNALsumamplitude_HB = new TH1F("h_averNOSIGNALsumamplitude_HB", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HE = new TH1F("h_averNOSIGNALsumamplitude_HE", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HF = new TH1F("h_averNOSIGNALsumamplitude_HF", " ", bac, 1., bac2);
    h_averNOSIGNALsumamplitude_HO = new TH1F("h_averNOSIGNALsumamplitude_HO", " ", bac, 1., bac2);

    // for channel SUM over depthes Amplitudes for each sub-detector
    h_sumamplitudechannel_HB = new TH1F("h_sumamplitudechannel_HB", " ", 100, 0., 2000.);
    h_sumamplitudechannel_HE = new TH1F("h_sumamplitudechannel_HE", " ", 100, 0., 3000.);
    h_sumamplitudechannel_HF = new TH1F("h_sumamplitudechannel_HF", " ", 100, 0., 7000.);
    h_sumamplitudechannel_HO = new TH1F("h_sumamplitudechannel_HO", " ", 100, 0., 10000.);

    // for event Amplitudes for each sub-detector
    h_eventamplitude_HB = new TH1F("h_eventamplitude_HB", " ", 100, 0., 80000.);
    h_eventamplitude_HE = new TH1F("h_eventamplitude_HE", " ", 100, 0., 100000.);
    h_eventamplitude_HF = new TH1F("h_eventamplitude_HF", " ", 100, 0., 150000.);
    h_eventamplitude_HO = new TH1F("h_eventamplitude_HO", " ", 100, 0., 250000.);

    // for event Occupancy for each sub-detector
    h_eventoccupancy_HB = new TH1F("h_eventoccupancy_HB", " ", 100, 0., 3000.);
    h_eventoccupancy_HE = new TH1F("h_eventoccupancy_HE", " ", 100, 0., 2000.);
    h_eventoccupancy_HF = new TH1F("h_eventoccupancy_HF", " ", 100, 0., 1000.);
    h_eventoccupancy_HO = new TH1F("h_eventoccupancy_HO", " ", 100, 0., 2500.);

    // for maxxSUMAmplitude
    h_maxxSUMAmpl_HB = new TH1F("h_maxxSUMAmpl_HB", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HE = new TH1F("h_maxxSUMAmpl_HE", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HF = new TH1F("h_maxxSUMAmpl_HF", " ", bac, 1., bac2);
    h_maxxSUMAmpl_HO = new TH1F("h_maxxSUMAmpl_HO", " ", bac, 1., bac2);

    // for maxxOCCUP
    h_maxxOCCUP_HB = new TH1F("h_maxxOCCUP_HB", " ", bac, 1., bac2);
    h_maxxOCCUP_HE = new TH1F("h_maxxOCCUP_HE", " ", bac, 1., bac2);
    h_maxxOCCUP_HF = new TH1F("h_maxxOCCUP_HF", " ", bac, 1., bac2);
    h_maxxOCCUP_HO = new TH1F("h_maxxOCCUP_HO", " ", bac, 1., bac2);
    //--------------------------------------------------
    // pedestals
    h_pedestal0_HB = new TH1F("h_pedestal0_HB", " ", 100, 0., 10.);
    h_pedestal1_HB = new TH1F("h_pedestal1_HB", " ", 100, 0., 10.);
    h_pedestal2_HB = new TH1F("h_pedestal2_HB", " ", 100, 0., 10.);
    h_pedestal3_HB = new TH1F("h_pedestal3_HB", " ", 100, 0., 10.);
    h_pedestalaver4_HB = new TH1F("h_pedestalaver4_HB", " ", 100, 0., 10.);
    h_pedestalaver9_HB = new TH1F("h_pedestalaver9_HB", " ", 100, 0., 10.);
    h_pedestalw0_HB = new TH1F("h_pedestalw0_HB", " ", 100, 0., 2.5);
    h_pedestalw1_HB = new TH1F("h_pedestalw1_HB", " ", 100, 0., 2.5);
    h_pedestalw2_HB = new TH1F("h_pedestalw2_HB", " ", 100, 0., 2.5);
    h_pedestalw3_HB = new TH1F("h_pedestalw3_HB", " ", 100, 0., 2.5);
    h_pedestalwaver4_HB = new TH1F("h_pedestalwaver4_HB", " ", 100, 0., 2.5);
    h_pedestalwaver9_HB = new TH1F("h_pedestalwaver9_HB", " ", 100, 0., 2.5);

    h_pedestal0_HE = new TH1F("h_pedestal0_HE", " ", 100, 0., 10.);
    h_pedestal1_HE = new TH1F("h_pedestal1_HE", " ", 100, 0., 10.);
    h_pedestal2_HE = new TH1F("h_pedestal2_HE", " ", 100, 0., 10.);
    h_pedestal3_HE = new TH1F("h_pedestal3_HE", " ", 100, 0., 10.);
    h_pedestalaver4_HE = new TH1F("h_pedestalaver4_HE", " ", 100, 0., 10.);
    h_pedestalaver9_HE = new TH1F("h_pedestalaver9_HE", " ", 100, 0., 10.);
    h_pedestalw0_HE = new TH1F("h_pedestalw0_HE", " ", 100, 0., 2.5);
    h_pedestalw1_HE = new TH1F("h_pedestalw1_HE", " ", 100, 0., 2.5);
    h_pedestalw2_HE = new TH1F("h_pedestalw2_HE", " ", 100, 0., 2.5);
    h_pedestalw3_HE = new TH1F("h_pedestalw3_HE", " ", 100, 0., 2.5);
    h_pedestalwaver4_HE = new TH1F("h_pedestalwaver4_HE", " ", 100, 0., 2.5);
    h_pedestalwaver9_HE = new TH1F("h_pedestalwaver9_HE", " ", 100, 0., 2.5);

    h_pedestal0_HF = new TH1F("h_pedestal0_HF", " ", 100, 0., 20.);
    h_pedestal1_HF = new TH1F("h_pedestal1_HF", " ", 100, 0., 20.);
    h_pedestal2_HF = new TH1F("h_pedestal2_HF", " ", 100, 0., 20.);
    h_pedestal3_HF = new TH1F("h_pedestal3_HF", " ", 100, 0., 20.);
    h_pedestalaver4_HF = new TH1F("h_pedestalaver4_HF", " ", 100, 0., 20.);
    h_pedestalaver9_HF = new TH1F("h_pedestalaver9_HF", " ", 100, 0., 20.);
    h_pedestalw0_HF = new TH1F("h_pedestalw0_HF", " ", 100, 0., 2.5);
    h_pedestalw1_HF = new TH1F("h_pedestalw1_HF", " ", 100, 0., 2.5);
    h_pedestalw2_HF = new TH1F("h_pedestalw2_HF", " ", 100, 0., 2.5);
    h_pedestalw3_HF = new TH1F("h_pedestalw3_HF", " ", 100, 0., 2.5);
    h_pedestalwaver4_HF = new TH1F("h_pedestalwaver4_HF", " ", 100, 0., 2.5);
    h_pedestalwaver9_HF = new TH1F("h_pedestalwaver9_HF", " ", 100, 0., 2.5);

    h_pedestal0_HO = new TH1F("h_pedestal0_HO", " ", 100, 0., 20.);
    h_pedestal1_HO = new TH1F("h_pedestal1_HO", " ", 100, 0., 20.);
    h_pedestal2_HO = new TH1F("h_pedestal2_HO", " ", 100, 0., 20.);
    h_pedestal3_HO = new TH1F("h_pedestal3_HO", " ", 100, 0., 20.);
    h_pedestalaver4_HO = new TH1F("h_pedestalaver4_HO", " ", 100, 0., 20.);
    h_pedestalaver9_HO = new TH1F("h_pedestalaver9_HO", " ", 100, 0., 20.);
    h_pedestalw0_HO = new TH1F("h_pedestalw0_HO", " ", 100, 0., 2.5);
    h_pedestalw1_HO = new TH1F("h_pedestalw1_HO", " ", 100, 0., 2.5);
    h_pedestalw2_HO = new TH1F("h_pedestalw2_HO", " ", 100, 0., 2.5);
    h_pedestalw3_HO = new TH1F("h_pedestalw3_HO", " ", 100, 0., 2.5);
    h_pedestalwaver4_HO = new TH1F("h_pedestalwaver4_HO", " ", 100, 0., 2.5);
    h_pedestalwaver9_HO = new TH1F("h_pedestalwaver9_HO", " ", 100, 0., 2.5);
    //--------------------------------------------------
    h_mapDepth1pedestalw_HB = new TH2F("h_mapDepth1pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HB = new TH2F("h_mapDepth2pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HB = new TH2F("h_mapDepth3pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HB = new TH2F("h_mapDepth4pedestalw_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestalw_HE = new TH2F("h_mapDepth1pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HE = new TH2F("h_mapDepth2pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HE = new TH2F("h_mapDepth3pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HE = new TH2F("h_mapDepth4pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5pedestalw_HE = new TH2F("h_mapDepth5pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6pedestalw_HE = new TH2F("h_mapDepth6pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7pedestalw_HE = new TH2F("h_mapDepth7pedestalw_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestalw_HF = new TH2F("h_mapDepth1pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestalw_HF = new TH2F("h_mapDepth2pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestalw_HF = new TH2F("h_mapDepth3pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HF = new TH2F("h_mapDepth4pedestalw_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestalw_HO = new TH2F("h_mapDepth4pedestalw_HO", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1pedestal_HB = new TH2F("h_mapDepth1pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HB = new TH2F("h_mapDepth2pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HB = new TH2F("h_mapDepth3pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HB = new TH2F("h_mapDepth4pedestal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestal_HE = new TH2F("h_mapDepth1pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HE = new TH2F("h_mapDepth2pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HE = new TH2F("h_mapDepth3pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HE = new TH2F("h_mapDepth4pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5pedestal_HE = new TH2F("h_mapDepth5pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6pedestal_HE = new TH2F("h_mapDepth6pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7pedestal_HE = new TH2F("h_mapDepth7pedestal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1pedestal_HF = new TH2F("h_mapDepth1pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2pedestal_HF = new TH2F("h_mapDepth2pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3pedestal_HF = new TH2F("h_mapDepth3pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HF = new TH2F("h_mapDepth4pedestal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4pedestal_HO = new TH2F("h_mapDepth4pedestal_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_pedestal00_HB = new TH1F("h_pedestal00_HB", " ", 100, 0., 10.);
    h_gain_HB = new TH1F("h_gain_HB", " ", 100, 0., 1.);
    h_respcorr_HB = new TH1F("h_respcorr_HB", " ", 100, 0., 2.5);
    h_timecorr_HB = new TH1F("h_timecorr_HB", " ", 100, 0., 30.);
    h_lutcorr_HB = new TH1F("h_lutcorr_HB", " ", 100, 0., 10.);
    h_difpedestal0_HB = new TH1F("h_difpedestal0_HB", " ", 100, -3., 3.);
    h_difpedestal1_HB = new TH1F("h_difpedestal1_HB", " ", 100, -3., 3.);
    h_difpedestal2_HB = new TH1F("h_difpedestal2_HB", " ", 100, -3., 3.);
    h_difpedestal3_HB = new TH1F("h_difpedestal3_HB", " ", 100, -3., 3.);

    h_pedestal00_HE = new TH1F("h_pedestal00_HE", " ", 100, 0., 10.);
    h_gain_HE = new TH1F("h_gain_HE", " ", 100, 0., 1.);
    h_respcorr_HE = new TH1F("h_respcorr_HE", " ", 100, 0., 2.5);
    h_timecorr_HE = new TH1F("h_timecorr_HE", " ", 100, 0., 30.);
    h_lutcorr_HE = new TH1F("h_lutcorr_HE", " ", 100, 0., 10.);

    h_pedestal00_HF = new TH1F("h_pedestal00_HF", " ", 100, 0., 10.);
    h_gain_HF = new TH1F("h_gain_HF", " ", 100, 0., 1.);
    h_respcorr_HF = new TH1F("h_respcorr_HF", " ", 100, 0., 2.5);
    h_timecorr_HF = new TH1F("h_timecorr_HF", " ", 100, 0., 30.);
    h_lutcorr_HF = new TH1F("h_lutcorr_HF", " ", 100, 0., 10.);

    h_pedestal00_HO = new TH1F("h_pedestal00_HO", " ", 100, 0., 10.);
    h_gain_HO = new TH1F("h_gain_HO", " ", 100, 0., 1.);
    h_respcorr_HO = new TH1F("h_respcorr_HO", " ", 100, 0., 2.5);
    h_timecorr_HO = new TH1F("h_timecorr_HO", " ", 100, 0., 30.);
    h_lutcorr_HO = new TH1F("h_lutcorr_HO", " ", 100, 0., 10.);
    //--------------------------------------------------
    float est6 = 2500.;
    int ist6 = 30;
    int ist2 = 60;
    h2_pedvsampl_HB = new TH2F("h2_pedvsampl_HB", " ", ist2, 0., 7.0, ist2, 0., est6);
    h2_pedwvsampl_HB = new TH2F("h2_pedwvsampl_HB", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HB = new TH1F("h_pedvsampl_HB", " ", ist6, 0., 7.0);
    h_pedwvsampl_HB = new TH1F("h_pedwvsampl_HB", " ", ist6, 0., 2.5);
    h_pedvsampl0_HB = new TH1F("h_pedvsampl0_HB", " ", ist6, 0., 7.);
    h_pedwvsampl0_HB = new TH1F("h_pedwvsampl0_HB", " ", ist6, 0., 2.5);
    h2_amplvsped_HB = new TH2F("h2_amplvsped_HB", " ", ist2, 0., est6, ist2, 0., 7.0);
    h2_amplvspedw_HB = new TH2F("h2_amplvspedw_HB", " ", ist2, 0., est6, ist2, 0., 2.5);
    h_amplvsped_HB = new TH1F("h_amplvsped_HB", " ", ist6, 0., est6);
    h_amplvspedw_HB = new TH1F("h_amplvspedw_HB", " ", ist6, 0., est6);
    h_amplvsped0_HB = new TH1F("h_amplvsped0_HB", " ", ist6, 0., est6);

    h2_pedvsampl_HE = new TH2F("h2_pedvsampl_HE", " ", ist2, 0., 7.0, ist2, 0., est6);
    h2_pedwvsampl_HE = new TH2F("h2_pedwvsampl_HE", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HE = new TH1F("h_pedvsampl_HE", " ", ist6, 0., 7.0);
    h_pedwvsampl_HE = new TH1F("h_pedwvsampl_HE", " ", ist6, 0., 2.5);
    h_pedvsampl0_HE = new TH1F("h_pedvsampl0_HE", " ", ist6, 0., 7.);
    h_pedwvsampl0_HE = new TH1F("h_pedwvsampl0_HE", " ", ist6, 0., 2.5);

    h2_pedvsampl_HF = new TH2F("h2_pedvsampl_HF", " ", ist2, 0., 20.0, ist2, 0., est6);
    h2_pedwvsampl_HF = new TH2F("h2_pedwvsampl_HF", " ", ist2, 0., 2.0, ist2, 0., est6);
    h_pedvsampl_HF = new TH1F("h_pedvsampl_HF", " ", ist6, 0., 20.0);
    h_pedwvsampl_HF = new TH1F("h_pedwvsampl_HF", " ", ist6, 0., 2.0);
    h_pedvsampl0_HF = new TH1F("h_pedvsampl0_HF", " ", ist6, 0., 20.);
    h_pedwvsampl0_HF = new TH1F("h_pedwvsampl0_HF", " ", ist6, 0., 2.0);

    h2_pedvsampl_HO = new TH2F("h2_pedvsampl_HO", " ", ist2, 0., 20.0, ist2, 0., est6);
    h2_pedwvsampl_HO = new TH2F("h2_pedwvsampl_HO", " ", ist2, 0., 2.5, ist2, 0., est6);
    h_pedvsampl_HO = new TH1F("h_pedvsampl_HO", " ", ist6, 0., 20.0);
    h_pedwvsampl_HO = new TH1F("h_pedwvsampl_HO", " ", ist6, 0., 2.5);
    h_pedvsampl0_HO = new TH1F("h_pedvsampl0_HO", " ", ist6, 0., 20.);
    h_pedwvsampl0_HO = new TH1F("h_pedwvsampl0_HO", " ", ist6, 0., 2.5);
    //--------------------------------------------------
    h_mapDepth1Ped0_HB = new TH2F("h_mapDepth1Ped0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HB = new TH2F("h_mapDepth1Ped1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HB = new TH2F("h_mapDepth1Ped2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HB = new TH2F("h_mapDepth1Ped3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HB = new TH2F("h_mapDepth1Pedw0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HB = new TH2F("h_mapDepth1Pedw1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HB = new TH2F("h_mapDepth1Pedw2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HB = new TH2F("h_mapDepth1Pedw3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HB = new TH2F("h_mapDepth2Ped0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HB = new TH2F("h_mapDepth2Ped1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HB = new TH2F("h_mapDepth2Ped2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HB = new TH2F("h_mapDepth2Ped3_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HB = new TH2F("h_mapDepth2Pedw0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HB = new TH2F("h_mapDepth2Pedw1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HB = new TH2F("h_mapDepth2Pedw2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HB = new TH2F("h_mapDepth2Pedw3_HB", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1Ped0_HE = new TH2F("h_mapDepth1Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HE = new TH2F("h_mapDepth1Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HE = new TH2F("h_mapDepth1Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HE = new TH2F("h_mapDepth1Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HE = new TH2F("h_mapDepth1Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HE = new TH2F("h_mapDepth1Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HE = new TH2F("h_mapDepth1Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HE = new TH2F("h_mapDepth1Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HE = new TH2F("h_mapDepth2Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HE = new TH2F("h_mapDepth2Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HE = new TH2F("h_mapDepth2Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HE = new TH2F("h_mapDepth2Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HE = new TH2F("h_mapDepth2Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HE = new TH2F("h_mapDepth2Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HE = new TH2F("h_mapDepth2Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HE = new TH2F("h_mapDepth2Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped0_HE = new TH2F("h_mapDepth3Ped0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped1_HE = new TH2F("h_mapDepth3Ped1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped2_HE = new TH2F("h_mapDepth3Ped2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Ped3_HE = new TH2F("h_mapDepth3Ped3_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw0_HE = new TH2F("h_mapDepth3Pedw0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw1_HE = new TH2F("h_mapDepth3Pedw1_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw2_HE = new TH2F("h_mapDepth3Pedw2_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3Pedw3_HE = new TH2F("h_mapDepth3Pedw3_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1Ped0_HF = new TH2F("h_mapDepth1Ped0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped1_HF = new TH2F("h_mapDepth1Ped1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped2_HF = new TH2F("h_mapDepth1Ped2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Ped3_HF = new TH2F("h_mapDepth1Ped3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw0_HF = new TH2F("h_mapDepth1Pedw0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw1_HF = new TH2F("h_mapDepth1Pedw1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw2_HF = new TH2F("h_mapDepth1Pedw2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1Pedw3_HF = new TH2F("h_mapDepth1Pedw3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped0_HF = new TH2F("h_mapDepth2Ped0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped1_HF = new TH2F("h_mapDepth2Ped1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped2_HF = new TH2F("h_mapDepth2Ped2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Ped3_HF = new TH2F("h_mapDepth2Ped3_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw0_HF = new TH2F("h_mapDepth2Pedw0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw1_HF = new TH2F("h_mapDepth2Pedw1_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw2_HF = new TH2F("h_mapDepth2Pedw2_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2Pedw3_HF = new TH2F("h_mapDepth2Pedw3_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth4Ped0_HO = new TH2F("h_mapDepth4Ped0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped1_HO = new TH2F("h_mapDepth4Ped1_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped2_HO = new TH2F("h_mapDepth4Ped2_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Ped3_HO = new TH2F("h_mapDepth4Ped3_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw0_HO = new TH2F("h_mapDepth4Pedw0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw1_HO = new TH2F("h_mapDepth4Pedw1_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw2_HO = new TH2F("h_mapDepth4Pedw2_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4Pedw3_HO = new TH2F("h_mapDepth4Pedw3_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_mapDepth1ADCAmpl12_HB = new TH2F("h_mapDepth1ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HB = new TH2F("h_mapDepth2ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HB = new TH2F("h_mapDepth3ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HB = new TH2F("h_mapDepth4ADCAmpl12_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl12_HE = new TH2F("h_mapDepth1ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HE = new TH2F("h_mapDepth2ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HE = new TH2F("h_mapDepth3ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HE = new TH2F("h_mapDepth4ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth5ADCAmpl12_HE = new TH2F("h_mapDepth5ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth6ADCAmpl12_HE = new TH2F("h_mapDepth6ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth7ADCAmpl12_HE = new TH2F("h_mapDepth7ADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth1ADCAmpl12SiPM_HE = new TH2F("h_mapDepth1ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12SiPM_HE = new TH2F("h_mapDepth2ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12SiPM_HE = new TH2F("h_mapDepth3ADCAmpl12SiPM_HE", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1ADCAmpl12_HF = new TH2F("h_mapDepth1ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2ADCAmpl12_HF = new TH2F("h_mapDepth2ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3ADCAmpl12_HF = new TH2F("h_mapDepth3ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth4ADCAmpl12_HF = new TH2F("h_mapDepth4ADCAmpl12_HF", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth4ADCAmpl12_HO = new TH2F("h_mapDepth4ADCAmpl12_HO", " ", neta, -41., 41., nphi, 0., bphi);

    h_mapDepth1linADCAmpl12_HE = new TH2F("h_mapDepth1linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth2linADCAmpl12_HE = new TH2F("h_mapDepth2linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapDepth3linADCAmpl12_HE = new TH2F("h_mapDepth3linADCAmpl12_HE", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_mapGetRMSOverNormalizedSignal_HB =
        new TH2F("h_mapGetRMSOverNormalizedSignal_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HB =
        new TH2F("h_mapGetRMSOverNormalizedSignal0_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HE =
        new TH2F("h_mapGetRMSOverNormalizedSignal_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HE =
        new TH2F("h_mapGetRMSOverNormalizedSignal0_HE", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HF =
        new TH2F("h_mapGetRMSOverNormalizedSignal_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HF =
        new TH2F("h_mapGetRMSOverNormalizedSignal0_HF", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal_HO =
        new TH2F("h_mapGetRMSOverNormalizedSignal_HO", " ", neta, -41., 41., nphi, 0., bphi);
    h_mapGetRMSOverNormalizedSignal0_HO =
        new TH2F("h_mapGetRMSOverNormalizedSignal0_HO", " ", neta, -41., 41., nphi, 0., bphi);
    //--------------------------------------------------
    h_shape_Ahigh_HB0 = new TH1F("h_shape_Ahigh_HB0", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB0 = new TH1F("h_shape0_Ahigh_HB0", " ", 10, 0., 10.);
    h_shape_Alow_HB0 = new TH1F("h_shape_Alow_HB0", " ", 10, 0., 10.);
    h_shape0_Alow_HB0 = new TH1F("h_shape0_Alow_HB0", " ", 10, 0., 10.);
    h_shape_Ahigh_HB1 = new TH1F("h_shape_Ahigh_HB1", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB1 = new TH1F("h_shape0_Ahigh_HB1", " ", 10, 0., 10.);
    h_shape_Alow_HB1 = new TH1F("h_shape_Alow_HB1", " ", 10, 0., 10.);
    h_shape0_Alow_HB1 = new TH1F("h_shape0_Alow_HB1", " ", 10, 0., 10.);
    h_shape_Ahigh_HB2 = new TH1F("h_shape_Ahigh_HB2", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB2 = new TH1F("h_shape0_Ahigh_HB2", " ", 10, 0., 10.);
    h_shape_Alow_HB2 = new TH1F("h_shape_Alow_HB2", " ", 10, 0., 10.);
    h_shape0_Alow_HB2 = new TH1F("h_shape0_Alow_HB2", " ", 10, 0., 10.);
    h_shape_Ahigh_HB3 = new TH1F("h_shape_Ahigh_HB3", " ", 10, 0., 10.);
    h_shape0_Ahigh_HB3 = new TH1F("h_shape0_Ahigh_HB3", " ", 10, 0., 10.);
    h_shape_Alow_HB3 = new TH1F("h_shape_Alow_HB3", " ", 10, 0., 10.);
    h_shape0_Alow_HB3 = new TH1F("h_shape0_Alow_HB3", " ", 10, 0., 10.);
    //--------------------------------------------------
    h_shape_bad_channels_HB = new TH1F("h_shape_bad_channels_HB", " ", 10, 0., 10.);
    h_shape0_bad_channels_HB = new TH1F("h_shape0_bad_channels_HB", " ", 10, 0., 10.);
    h_shape_good_channels_HB = new TH1F("h_shape_good_channels_HB", " ", 10, 0., 10.);
    h_shape0_good_channels_HB = new TH1F("h_shape0_good_channels_HB", " ", 10, 0., 10.);
    h_shape_bad_channels_HE = new TH1F("h_shape_bad_channels_HE", " ", 10, 0., 10.);
    h_shape0_bad_channels_HE = new TH1F("h_shape0_bad_channels_HE", " ", 10, 0., 10.);
    h_shape_good_channels_HE = new TH1F("h_shape_good_channels_HE", " ", 10, 0., 10.);
    h_shape0_good_channels_HE = new TH1F("h_shape0_good_channels_HE", " ", 10, 0., 10.);
    h_shape_bad_channels_HF = new TH1F("h_shape_bad_channels_HF", " ", 10, 0., 10.);
    h_shape0_bad_channels_HF = new TH1F("h_shape0_bad_channels_HF", " ", 10, 0., 10.);
    h_shape_good_channels_HF = new TH1F("h_shape_good_channels_HF", " ", 10, 0., 10.);
    h_shape0_good_channels_HF = new TH1F("h_shape0_good_channels_HF", " ", 10, 0., 10.);
    h_shape_bad_channels_HO = new TH1F("h_shape_bad_channels_HO", " ", 10, 0., 10.);
    h_shape0_bad_channels_HO = new TH1F("h_shape0_bad_channels_HO", " ", 10, 0., 10.);
    h_shape_good_channels_HO = new TH1F("h_shape_good_channels_HO", " ", 10, 0., 10.);
    h_shape0_good_channels_HO = new TH1F("h_shape0_good_channels_HO", " ", 10, 0., 10.);
    //--------------------------------------------------
    //    if(flagcpuoptimization_== 0 ) {

    int spl = 1000;
    float spls = 5000;
    h_sumamplitude_depth1_HB = new TH1F("h_sumamplitude_depth1_HB", " ", spl, 0., spls);
    h_sumamplitude_depth2_HB = new TH1F("h_sumamplitude_depth2_HB", " ", spl, 0., spls);
    h_sumamplitude_depth1_HE = new TH1F("h_sumamplitude_depth1_HE", " ", spl, 0., spls);
    h_sumamplitude_depth2_HE = new TH1F("h_sumamplitude_depth2_HE", " ", spl, 0., spls);
    h_sumamplitude_depth3_HE = new TH1F("h_sumamplitude_depth3_HE", " ", spl, 0., spls);
    h_sumamplitude_depth1_HF = new TH1F("h_sumamplitude_depth1_HF", " ", spl, 0., spls);
    h_sumamplitude_depth2_HF = new TH1F("h_sumamplitude_depth2_HF", " ", spl, 0., spls);
    h_sumamplitude_depth4_HO = new TH1F("h_sumamplitude_depth4_HO", " ", spl, 0., spls);
    int spl0 = 1000;
    float spls0 = 10000;
    h_sumamplitude_depth1_HB0 = new TH1F("h_sumamplitude_depth1_HB0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HB0 = new TH1F("h_sumamplitude_depth2_HB0", " ", spl0, 0., spls0);
    h_sumamplitude_depth1_HE0 = new TH1F("h_sumamplitude_depth1_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HE0 = new TH1F("h_sumamplitude_depth2_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth3_HE0 = new TH1F("h_sumamplitude_depth3_HE0", " ", spl0, 0., spls0);
    h_sumamplitude_depth1_HF0 = new TH1F("h_sumamplitude_depth1_HF0", " ", spl0, 0., spls0);
    h_sumamplitude_depth2_HF0 = new TH1F("h_sumamplitude_depth2_HF0", " ", spl0, 0., spls0);
    h_sumamplitude_depth4_HO0 = new TH1F("h_sumamplitude_depth4_HO0", " ", spl0, 0., spls0);
    int spl1 = 1000;
    float spls1 = 100000;
    h_sumamplitude_depth1_HB1 = new TH1F("h_sumamplitude_depth1_HB1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HB1 = new TH1F("h_sumamplitude_depth2_HB1", " ", spl1, 0., spls1);
    h_sumamplitude_depth1_HE1 = new TH1F("h_sumamplitude_depth1_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HE1 = new TH1F("h_sumamplitude_depth2_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth3_HE1 = new TH1F("h_sumamplitude_depth3_HE1", " ", spl1, 0., spls1);
    h_sumamplitude_depth1_HF1 = new TH1F("h_sumamplitude_depth1_HF1", " ", spl1, 0., spls1);
    h_sumamplitude_depth2_HF1 = new TH1F("h_sumamplitude_depth2_HF1", " ", spl1, 0., spls1);
    h_sumamplitude_depth4_HO1 = new TH1F("h_sumamplitude_depth4_HO1", " ", spl1, 0., spls1);

    h_Amplitude_forCapIdErrors_HB1 = new TH1F("h_Amplitude_forCapIdErrors_HB1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HB2 = new TH1F("h_Amplitude_forCapIdErrors_HB2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE1 = new TH1F("h_Amplitude_forCapIdErrors_HE1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE2 = new TH1F("h_Amplitude_forCapIdErrors_HE2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HE3 = new TH1F("h_Amplitude_forCapIdErrors_HE3", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HF1 = new TH1F("h_Amplitude_forCapIdErrors_HF1", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HF2 = new TH1F("h_Amplitude_forCapIdErrors_HF2", " ", 100, 0., 30000.);
    h_Amplitude_forCapIdErrors_HO4 = new TH1F("h_Amplitude_forCapIdErrors_HO4", " ", 100, 0., 30000.);

    h_Amplitude_notCapIdErrors_HB1 = new TH1F("h_Amplitude_notCapIdErrors_HB1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HB2 = new TH1F("h_Amplitude_notCapIdErrors_HB2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE1 = new TH1F("h_Amplitude_notCapIdErrors_HE1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE2 = new TH1F("h_Amplitude_notCapIdErrors_HE2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HE3 = new TH1F("h_Amplitude_notCapIdErrors_HE3", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HF1 = new TH1F("h_Amplitude_notCapIdErrors_HF1", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HF2 = new TH1F("h_Amplitude_notCapIdErrors_HF2", " ", 100, 0., 30000.);
    h_Amplitude_notCapIdErrors_HO4 = new TH1F("h_Amplitude_notCapIdErrors_HO4", " ", 100, 0., 30000.);

    h_2DAtaildepth1_HB = new TH2F("h_2DAtaildepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0Ataildepth1_HB = new TH2F("h_2D0Ataildepth1_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2DAtaildepth2_HB = new TH2F("h_2DAtaildepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);
    h_2D0Ataildepth2_HB = new TH2F("h_2D0Ataildepth2_HB", " ", neta, -41., 41., nphi, 0., bphi);

    ////////////////////////////////////////////////////////////////////////////////////
  }  //if(recordHistoes_
  if (verbosity > 0)
    cout << "========================   booking DONE   +++++++++++++++++++++++++++" << endl;
  ///////////////////////////////////////////////////////            ntuples:
  if (recordNtuples_) {
    myTree = new TTree("Hcal", "Hcal Tree");
    myTree->Branch("Nevent", &Nevent, "Nevent/I");
    myTree->Branch("Run", &Run, "Run/I");

  }  //if(recordNtuples_
  if (verbosity > 0)
    cout << "========================   beignJob  finish   +++++++++++++++++++++++++++" << endl;
  //////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrors(HBHEDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  //    double tool[100];
  if (verbosity == -22)
    std::cout << "**************   in loop over Digis   counter =     " << nnnnnnhbhe << std::endl;
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int sumcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;
  int TSsize = 10;
  TSsize = digiItr->size();

  ///////////////////////////////////////
  for (int ii = 0; ii < TSsize; ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;

    if (ii == 0)
      firstcapid = capid;
    sumcapid += capid;

    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }

  }  // for

  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;

  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Energy
  // Energy:

  double ampl = 0.;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }

  ///////////////////////////////////////Digis
  // Digis:
  // HB
  if (sub == 1) {
    h_errorGeneral_HB->Fill(double(errorGeneral), 1.);
    h_error1_HB->Fill(double(error1), 1.);
    h_error2_HB->Fill(double(error2), 1.);
    h_error3_HB->Fill(double(error3), 1.);
    h_error4_HB->Fill(double(error4), 1.);
    h_error5_HB->Fill(double(error5), 1.);
    h_error6_HB->Fill(double(error6), 1.);
    h_error7_HB->Fill(double(error7), 1.);
    h_repetedcapid_HB->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HB->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HB->Fill(double(ieta), double(iphi));
      h_fiber0_HB->Fill(double(ERRORfiber), 1.);
      h_fiber1_HB->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HB->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HB->Fill(ampl, 1.);
    }
  }
  // HE
  if (sub == 2) {
    h_errorGeneral_HE->Fill(double(errorGeneral), 1.);
    h_error1_HE->Fill(double(error1), 1.);
    h_error2_HE->Fill(double(error2), 1.);
    h_error3_HE->Fill(double(error3), 1.);
    h_error4_HE->Fill(double(error4), 1.);
    h_error5_HE->Fill(double(error5), 1.);
    h_error6_HE->Fill(double(error6), 1.);
    h_error7_HE->Fill(double(error7), 1.);
    h_repetedcapid_HE->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HE->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HE->Fill(double(ieta), double(iphi));
      h_fiber0_HE->Fill(double(ERRORfiber), 1.);
      h_fiber1_HE->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HE->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HE->Fill(ampl, 1.);
    }
  }
  //    ha2->Fill(double(ieta), double(iphi));
}  //fillDigiErrors
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHBHEQIE11
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsQIE11(QIE11DataFrame qie11df) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  DetId detid = qie11df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int error1 = 0;
  // !!!!!!
  bool anycapid = true;
  //    bool anyer      =  false;
  //    bool anydv      =  true;
  // for help:
  int firstcapid = 0;
  int sumcapid = 0;
  int lastcapid = 0, capid = 0;
  int repetedcapid = 0;
  // loop over the samples in the digi
  nTS = qie11df.samples();
  ///////////////////////////////////////
  for (int ii = 0; ii < nTS; ii++) {
    capid = qie11df[ii].capid();  // capId (0-3, sequential)
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
    sumcapid += capid;
  }  // for
  ///////////////////////////////////////
  if (!anycapid)
    error1 = 1;
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  double ampl = 0.;
  for (int ii = 0; ii < nTS; ii++) {
    int adc = qie11df[ii].adc();

    double ampldefault = adc2fC_QIE11_shunt6[adc];
    if (flaguseshunt_ == 1)
      ampldefault = adc2fC_QIE11_shunt1[adc];

    ampl += ampldefault;  //
  }
  ///////////////////////////////////////Digis
  // Digis:HBHE
  if (sub == 1) {
    h_error1_HB->Fill(double(error1), 1.);
    h_repetedcapid_HB->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HB->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HB->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HB->Fill(double(ieta), double(iphi));
      h_errorGeneral_HB->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HB->Fill(ampl, 1.);
    }
  }
  if (sub == 2) {
    h_error1_HE->Fill(double(error1), 1.);
    h_repetedcapid_HE->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HE->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 5)
        h_mapDepth5Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 6)
        h_mapDepth6Error_HE->Fill(double(ieta), double(iphi));
      if (mdepth == 7)
        h_mapDepth7Error_HE->Fill(double(ieta), double(iphi));
      h_errorGeneral_HE->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HE->Fill(ampl, 1.);
    }
  }
}  //fillDigiErrorsQIE11
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHF
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHF(HFDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  if (mdepth > 2)
    std::cout << " HF DIGI ??????????????   ERROR       mdepth =  " << mdepth << std::endl;
  // !!!!!!
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int sumcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;

  int TSsize = 10;
  TSsize = digiItr->size();
  ///////////////////////////////////////
  for (int ii = 0; ii < TSsize; ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
    sumcapid += capid;
    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
  }  // for
  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;
  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Ampl
  double ampl = 0.;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }
  ///////////////////////////////////////Digis
  // Digis: HF
  if (sub == 4) {
    h_errorGeneral_HF->Fill(double(errorGeneral), 1.);
    h_error1_HF->Fill(double(error1), 1.);
    h_error2_HF->Fill(double(error2), 1.);
    h_error3_HF->Fill(double(error3), 1.);
    h_error4_HF->Fill(double(error4), 1.);
    h_error5_HF->Fill(double(error5), 1.);
    h_error6_HF->Fill(double(error6), 1.);
    h_error7_HF->Fill(double(error7), 1.);
    h_repetedcapid_HF->Fill(double(repetedcapid), 1.);
    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HF->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HF->Fill(double(ieta), double(iphi));
      h_fiber0_HF->Fill(double(ERRORfiber), 1.);
      h_fiber1_HF->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HF->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HF->Fill(ampl, 1.);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    fillDigiErrorsHFQIE10
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHFQIE10(QIE10DataFrame qie10df) {
  CaloSamples toolOriginal;  // TS
  //  double tool[100];
  DetId detid = qie10df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  // !!!!!!
  int error1 = 0;
  // !!!!!!
  bool anycapid = true;
  //    bool anyer      =  false;
  //    bool anydv      =  true;
  // for help:
  int firstcapid = 0;
  int sumcapid = 0;
  int lastcapid = 0, capid = 0;
  int repetedcapid = 0;
  // loop over the samples in the digi
  nTS = qie10df.samples();
  ///////////////////////////////////////
  for (int ii = 0; ii < nTS; ii++) {
    capid = qie10df[ii].capid();  // capId (0-3, sequential)
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;
    if (ii == 0)
      firstcapid = capid;
    sumcapid += capid;
  }  // for
  ///////////////////////////////////////
  if (!anycapid)
    error1 = 1;
  //    if( anyer )                         error2 = 1;
  //    if( !anydv )                        error3 = 1;
  ///////////////////////////////////////Energy
  // Energy:
  // int adc = qie10df[ii].adc();
  // int tdc = qie10df[ii].le_tdc();
  // int trail = qie10df[ii].te_tdc();
  // int capid = qie10df[ii].capid();
  // int soi = qie10df[ii].soi();
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // store pulse information
  // THIS NEEDS TO BE UPDATED AND IS ONLY
  // BEING USED AS A PLACE HOLDER UNTIL THE
  // REAL LINEARIZATION CONSTANTS ARE DEFINED
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  double ampl = 0.;
  for (int ii = 0; ii < nTS; ii++) {
    int adc = qie10df[ii].adc();
    double ampldefault = adc2fC_QIE10[adc];
    ampl += ampldefault;  //
  }
  ///////////////////////////////////////Digis
  // Digis:HF
  if (sub == 4) {
    h_error1_HF->Fill(double(error1), 1.);
    h_repetedcapid_HF->Fill(double(repetedcapid), 1.);
    if (error1 != 0) {
      //      if(error1 !=0 || error2 !=0 || error3 !=0 ) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HF->Fill(ampl, 1.);
      if (mdepth == 1)
        h_mapDepth1Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 2)
        h_mapDepth2Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 3)
        h_mapDepth3Error_HF->Fill(double(ieta), double(iphi));
      if (mdepth == 4)
        h_mapDepth4Error_HF->Fill(double(ieta), double(iphi));
      h_errorGeneral_HF->Fill(double(firstcapid), 1.);
    } else {
      h_amplFine_HF->Fill(ampl, 1.);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ------------ method called for each event  ------------
void CMTRawAnalyzer::fillDigiErrorsHO(HODigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;
  int sub = cell.subdet();  // 1-HB, 2-HE, 3-HO, 4-HF
  int errorGeneral = 0;
  int error1 = 0;
  int error2 = 0;
  int error3 = 0;
  int error4 = 0;
  int error5 = 0;
  int error6 = 0;
  int error7 = 0;
  // !!!!!!
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  // for help:
  int firstcapid = 0;
  int sumcapid = 0;
  int lastcapid = 0, capid = 0;
  int ERRORfiber = -10;
  int ERRORfiberChan = -10;
  int ERRORfiberAndChan = -10;
  int repetedcapid = 0;
  for (int ii = 0; ii < (*digiItr).size(); ii++) {
    capid = (*digiItr)[ii].capid();                    // capId (0-3, sequential)
    bool er = (*digiItr)[ii].er();                     // error
    bool dv = (*digiItr)[ii].dv();                     // valid data
    int fiber = (*digiItr)[ii].fiber();                // get the fiber number
    int fiberChan = (*digiItr)[ii].fiberChan();        // get the fiber channel number
    int fiberAndChan = (*digiItr)[ii].fiberAndChan();  // get the id channel
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
      if (capid != lastcapid) {
      } else {
        repetedcapid = 1;
      }
    }
    lastcapid = capid;

    if (ii == 0)
      firstcapid = capid;
    sumcapid += capid;

    if (er) {
      anyer = true;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }
    if (!dv) {
      anydv = false;
      ERRORfiber = fiber;
      ERRORfiberChan = fiberChan;
      ERRORfiberAndChan = fiberAndChan;
    }

  }  // for

  ///////////////////////////////////////
  if (firstcapid == 0 && !anycapid)
    errorGeneral = 1;
  if (firstcapid == 1 && !anycapid)
    errorGeneral = 2;
  if (firstcapid == 2 && !anycapid)
    errorGeneral = 3;
  if (firstcapid == 3 && !anycapid)
    errorGeneral = 4;
  if (!anycapid)
    error1 = 1;
  if (anyer)
    error2 = 1;
  if (!anydv)
    error3 = 1;

  if (!anycapid && anyer)
    error4 = 1;
  if (!anycapid && !anydv)
    error5 = 1;
  if (!anycapid && anyer && !anydv)
    error6 = 1;
  if (anyer && !anydv)
    error7 = 1;
  ///////////////////////////////////////Energy
  // Energy:
  double ampl = 0.;
  for (int ii = 0; ii < (*digiItr).size(); ii++) {
    double ampldefault = adc2fC[digiItr->sample(ii).adc()];
    ampl += ampldefault;  // fC
  }
  ///////////////////////////////////////Digis
  // Digis:
  // HO
  if (sub == 3) {
    h_errorGeneral_HO->Fill(double(errorGeneral), 1.);
    h_error1_HO->Fill(double(error1), 1.);
    h_error2_HO->Fill(double(error2), 1.);
    h_error3_HO->Fill(double(error3), 1.);
    h_error4_HO->Fill(double(error4), 1.);
    h_error5_HO->Fill(double(error5), 1.);
    h_error6_HO->Fill(double(error6), 1.);
    h_error7_HO->Fill(double(error7), 1.);
    h_repetedcapid_HO->Fill(double(repetedcapid), 1.);

    if (error1 != 0 || error2 != 0 || error3 != 0) {
      if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 0)
        ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
      h_amplError_HO->Fill(ampl, 1.);
      if (mdepth == 4)
        h_mapDepth4Error_HO->Fill(double(ieta), double(iphi));
      // to be divided by h_mapDepth4_HO

      if (mdepth != 4)
        std::cout << " mdepth HO = " << mdepth << std::endl;
      h_fiber0_HO->Fill(double(ERRORfiber), 1.);
      h_fiber1_HO->Fill(double(ERRORfiberChan), 1.);
      h_fiber2_HO->Fill(double(ERRORfiberAndChan), 1.);
    } else {
      h_amplFine_HO->Fill(ampl, 1.);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitude(HBHEDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  double toolwithPedSubtr[100];        // TS
  double lintoolwithoutPedSubtr[100];  // TS
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta0 = cell.ieta();     //-41 +41 !=0
  int ieta = ieta0;
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // 1-HB, 2-HE (HFDigiCollection: 4-HF)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  //    const HcalGainWidth* gainWidth = conditions->getGainWidth(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  //    HcalCalibrations calib = conditions->getHcalCalibrations(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double difpedestal0 = 0.;
  double difpedestal1 = 0.;
  double difpedestal2 = 0.;
  double difpedestal3 = 0.;

  double amplitudewithPedSubtr1 = 0.;
  double amplitudewithPedSubtr2 = 0.;
  double amplitudewithPedSubtr3 = 0.;
  double amplitudewithPedSubtr4 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double amplitude345 = 0.;
  double ampl = 0.;
  double linamplitudewithoutPedSubtr = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  //    int TSsize = 10;
  int TSsize = 10;
  //     if((*digiItr).size() !=  10) std::cout << "TSsize HBHE != 10 and = " <<(*digiItr).size()<< std::endl;
  if ((*digiItr).size() != TSsize)
    errorBtype = 1.;
  TSsize = digiItr->size();
  //     ii = 0 to 9
  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<digiItr->size(); ii++) {
    double ampldefaultwithPedSubtr = 0.;
    double linampldefaultwithoutPedSubtr = 0.;
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive ADCcounts
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts linearized
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    ampldefaultwithPedSubtr = ampldefault0;
    linampldefaultwithoutPedSubtr = ampldefault2;

    int capid = ((*digiItr)[ii]).capid();
    //      double pedestal = calib.pedestal(capid);
    double pedestalINI = pedestal00->getValue(capid);
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    ampldefaultwithPedSubtr -= pedestal;  // pedestal subtraction
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    //      ampldefault*= calib.respcorrgain(capid) ; // fC --> GeV
    tool[ii] = ampldefault;
    toolwithPedSubtr[ii] = ampldefaultwithPedSubtr;
    lintoolwithoutPedSubtr[ii] = linampldefaultwithoutPedSubtr;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal0 = pedestal - pedestalINI;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal1 = pedestal - pedestalINI;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal2 = pedestal - pedestalINI;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal3 = pedestal - pedestalINI;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    ///   for choice TSs, raddam only:
    //     TS = 1 to 10:  1  2  3  4  5  6  7  8  9  10
    //     ii = 0 to  9:  0  1  2  3  4  5  6  7  8   9
    //     var.1             ----------------------
    //     var.2                ----------------
    //     var.3                   ----------
    //     var.4                   -------
    //
    // TS = 2-9      for raddam only  var.1
    if (ii > 0 && ii < 9)
      amplitudewithPedSubtr1 += ampldefaultwithPedSubtr;  //
    // TS = 3-8      for raddam only  var.2
    if (ii > 1 && ii < 8)
      amplitudewithPedSubtr2 += ampldefaultwithPedSubtr;  //
    // TS = 4-7      for raddam only  var.3
    if (ii > 2 && ii < 7)
      amplitudewithPedSubtr3 += ampldefaultwithPedSubtr;  //
    // TS = 4-6      for raddam only  var.4
    if (ii > 2 && ii < 6)
      amplitudewithPedSubtr4 += ampldefaultwithPedSubtr;  //
    //
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //

    if (ii == 3 || ii == 4 || ii == 5)
      amplitude345 += ampldefault;
    if (flagcpuoptimization_ == 0) {
      //////
    }  //flagcpuoptimization
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }                                                                     //for 1
  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71  HBHE
  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);
  if (ts_with_max_signal > -1 && ts_with_max_signal < 10)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < 10)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < 10)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < 10)
    ampl += tool[ts_with_max_signal - 1];

  ///----------------------------------------------------------------------------------------------------  for raddam:
  if (ts_with_max_signal > -1 && ts_with_max_signal < 10)
    linamplitudewithoutPedSubtr = lintoolwithoutPedSubtr[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < 10)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < 10)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < 10)
    linamplitudewithoutPedSubtr += lintoolwithoutPedSubtr[ts_with_max_signal - 1];

  double ratio = 0.;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1
  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    //    std::cout << " ii = " << ii  << " capid = " << capid  << " ((lastcapid+1)%4) = " << ((lastcapid+1)%4)  << std::endl;
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2
  if (!anycapid || anyer || !anydv)
    error = 1;

  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9

  // CapIdErrors end  /////////////////////////////////////////////////////////

  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_forCapIdErrors_HE3->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_notCapIdErrors_HE3->Fill(amplitude, 1.);
  }

  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<10; ii++) {
    double ampldefault = tool[ii];
    ///
    if (sub == 1) {
      if (amplitude > 120) {
        h_shape_Ahigh_HB0->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB0->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB0->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB0->Fill(float(ii), 1.);
      }  //HB0
      ///
      if (pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        h_shape_Ahigh_HB1->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB1->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB1->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB1->Fill(float(ii), 1.);
      }  //HB1
      if (error == 0) {
        h_shape_Ahigh_HB2->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB2->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB2->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB2->Fill(float(ii), 1.);
      }  //HB2
      ///
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        h_shape_Ahigh_HB3->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB3->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB3->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB3->Fill(float(ii), 1.);
      }  //HB3

    }  // sub   HB

  }  //for 3 over TSs

  if (sub == 1) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_ || rmsamplitude < rmsHBMin_ ||
        rmsamplitude > rmsHBMax_ || pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ ||
        pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestalw0 < pedestalwHBMax_ ||
        pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ || pedestalw3 < pedestalwHBMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HB->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HB->Fill(float(ii), 1.);
      }
    }
  }  // sub   HB
  if (sub == 2) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_ || rmsamplitude < rmsHEMin_ ||
        rmsamplitude > rmsHEMax_ || pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ ||
        pedestal2 < pedestalHEMax_ || pedestal3 < pedestalHEMax_ || pedestalw0 < pedestalwHEMax_ ||
        pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ || pedestalw3 < pedestalwHEMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HE->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HE->Fill(float(ii), 1.);
      }
    }
  }  // sub   HE

  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HB
  if (sub == 1) {
    if (studyPedestalCorrelations_) {
      //   //   //   //   //   //   //   //   //  HB       PedestalCorrelations :
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HB->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl_HB->Fill(mypedestal, amplitude);
      h_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HB->Fill(mypedestal, 1.);
      h_pedwvsampl0_HB->Fill(mypedestalw, 1.);

      h2_amplvsped_HB->Fill(amplitude, mypedestal);
      h2_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped_HB->Fill(amplitude, mypedestal);
      h_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped0_HB->Fill(amplitude, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HB->Fill(pedestal0, 1.);
      h_pedestal1_HB->Fill(pedestal1, 1.);
      h_pedestal2_HB->Fill(pedestal2, 1.);
      h_pedestal3_HB->Fill(pedestal3, 1.);
      h_pedestalaver4_HB->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HB->Fill(pedestalaver9, 1.);
      h_pedestalw0_HB->Fill(pedestalw0, 1.);
      h_pedestalw1_HB->Fill(pedestalw1, 1.);
      h_pedestalw2_HB->Fill(pedestalw2, 1.);
      h_pedestalw3_HB->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HB->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HB->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HB->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HB->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HB->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HB->Fill(respcorr->getValue(), 1.);
      h_timecorr_HB->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HB->Fill(lutcorr->getValue(), 1.);
      h_difpedestal0_HB->Fill(difpedestal0, 1.);
      h_difpedestal1_HB->Fill(difpedestal1, 1.);
      h_difpedestal2_HB->Fill(difpedestal2, 1.);
      h_difpedestal3_HB->Fill(difpedestal3, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345Zoom1_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345_HB->Fill(amplitude345, 1.);
      if (error == 0) {
        h_ADCAmpl_HBCapIdNoError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdNoError->Fill(amplitude345, 1.);
      }
      if (error == 1) {
        h_ADCAmpl_HBCapIdError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdError->Fill(amplitude345, 1.);
      }
      h_ADCAmplZoom_HB->Fill(amplitude, 1.);
      h_ADCAmplZoom1_HB->Fill(amplitude, 1.);
      h_ADCAmpl_HB->Fill(amplitude, 1.);

      h_AmplitudeHBrest->Fill(amplitude, 1.);
      h_AmplitudeHBrest1->Fill(amplitude, 1.);
      h_AmplitudeHBrest6->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >400.) averSIGNALoccupancy_HB += 1.;
      if (amplitude < 35.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      h_bcnvsamplitude_HB->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HB->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HB->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HB->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HB->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHBMin_ || aveamplitude1 > TSmeanHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HB->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHBMin_ || ts_with_max_signal > TSpeakHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HB->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHBMin_ || rmsamplitude > rmsHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HB->Fill(ratio, 1.);
      if (ratio < ratioHBMin_ || ratio > ratioHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        // //
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HB All
    if (mdepth == 1)
      h_mapDepth1_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HB->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 1 )

  // HE
  if (sub == 2) {
    //   //   //   //   //   //   //   //   //  HE       PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HE->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl_HE->Fill(mypedestal, amplitude);
      h_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HE->Fill(mypedestal, 1.);
      h_pedwvsampl0_HE->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HE->Fill(pedestal0, 1.);
      h_pedestal1_HE->Fill(pedestal1, 1.);
      h_pedestal2_HE->Fill(pedestal2, 1.);
      h_pedestal3_HE->Fill(pedestal3, 1.);
      h_pedestalaver4_HE->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HE->Fill(pedestalaver9, 1.);
      h_pedestalw0_HE->Fill(pedestalw0, 1.);
      h_pedestalw1_HE->Fill(pedestalw1, 1.);
      h_pedestalw2_HE->Fill(pedestalw2, 1.);
      h_pedestalw3_HE->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HE->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HE->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 3) {
        h_mapDepth3Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth3Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth3Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth3Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth3Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth3Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth3Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth3Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHEMax_ || pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ ||
          pedestalw3 < pedestalwHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ || pedestal2 < pedestalHEMax_ ||
          pedestal3 < pedestalHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HE->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HE->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HE->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HE->Fill(respcorr->getValue(), 1.);
      h_timecorr_HE->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HE->Fill(lutcorr->getValue(), 1.);
    }  //

    //     h_mapDepth1ADCAmpl12SiPM_HE
    //   //   //   //   //   //   //   //   //  HE       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HE->Fill(ampl, 1.);
      h_ADCAmpl345Zoom1_HE->Fill(amplitude345, 1.);
      h_ADCAmpl345_HE->Fill(amplitude345, 1.);
      h_ADCAmpl_HE->Fill(amplitude, 1.);

      h_ADCAmplrest_HE->Fill(amplitude, 1.);
      h_ADCAmplrest1_HE->Fill(amplitude, 1.);
      h_ADCAmplrest6_HE->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HE->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude > 700.) averSIGNALoccupancy_HE += 1.;
      if (amplitude < 500.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);

      if (mdepth == 1) {
        h_mapDepth1ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth1linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }
      if (mdepth == 2) {
        h_mapDepth2ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth2linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }
      if (mdepth == 3) {
        h_mapDepth3ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapDepth3linADCAmpl12_HE->Fill(double(ieta), double(iphi), linamplitudewithoutPedSubtr);
      }

      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HE->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HE->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HE->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HE->Fill(float(orbitNum), 1.);

      ////////////////////////////////////////////////////////////////////////////////////////////////////////////

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;

    }  //if(studyADCAmplHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HE->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHEMin_ || aveamplitude1 > TSmeanHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HE->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHEMin_ || ts_with_max_signal > TSpeakHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HE->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHEMin_ || rmsamplitude > rmsHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HE       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HE->Fill(ratio, 1.);
      if (ratio < ratioHEMin_ || ratio > ratioHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE       DiffAmplitude:
    if (studyDiffAmplHist_) {
      // gain stability:
      if (mdepth == 1)
        h_mapDepth1AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);

    }  // if(studyDiffAmplHist_)

    // RADDAM filling:
    if (flagLaserRaddam_ > 0) {
      double amplitudewithPedSubtr = 0.;

      //for cut on A_channel:
      if (ts_with_max_signal > -1 && ts_with_max_signal < 10)
        amplitudewithPedSubtr = toolwithPedSubtr[ts_with_max_signal];
      if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < 10)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal + 2];
      if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < 10)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal + 1];
      if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < 10)
        amplitudewithPedSubtr += toolwithPedSubtr[ts_with_max_signal - 1];

      h_AamplitudewithPedSubtr_RADDAM_HE->Fill(amplitudewithPedSubtr);
      h_AamplitudewithPedSubtr_RADDAM_HEzoom0->Fill(amplitudewithPedSubtr);
      h_AamplitudewithPedSubtr_RADDAM_HEzoom1->Fill(amplitudewithPedSubtr);

      if (amplitudewithPedSubtr > 50.) {
        if (flagLaserRaddam_ > 1) {
          mapRADDAM_HE[mdepth - 1][ieta + 41][iphi] += amplitudewithPedSubtr;
          ++mapRADDAM0_HE[mdepth - 1][ieta + 41][iphi];
        }

        if (mdepth == 1) {
          h_mapDepth1RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth1RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth1RADDAM_HE->Fill(amplitudewithPedSubtr);
        }
        if (mdepth == 2) {
          h_mapDepth2RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth2RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth2RADDAM_HE->Fill(amplitudewithPedSubtr);
        }
        if (mdepth == 3) {
          h_mapDepth3RADDAM_HE->Fill(double(ieta), double(iphi), amplitudewithPedSubtr);
          h_mapDepth3RADDAM0_HE->Fill(double(ieta), double(iphi), 1.);
          h_A_Depth3RADDAM_HE->Fill(amplitudewithPedSubtr);
        }

        // (d1 & eta 17-29)                       L1
        int LLLLLL111111 = 0;
        if ((mdepth == 1 && fabs(ieta0) > 16 && fabs(ieta0) < 30))
          LLLLLL111111 = 1;
        // (d2 & eta 17-26) && (d3 & eta 27-28)   L2
        int LLLLLL222222 = 0;
        if ((mdepth == 2 && fabs(ieta0) > 16 && fabs(ieta0) < 27) ||
            (mdepth == 3 && fabs(ieta0) > 26 && fabs(ieta0) < 29))
          LLLLLL222222 = 1;
        //
        if (LLLLLL111111 == 1) {
          //forStudy	    h_mapLayer1RADDAM_HE->Fill(fabs(double(ieta0)), amplitudewithPedSubtr); h_mapLayer1RADDAM0_HE->Fill(fabs(double(ieta0)), 1.); h_A_Layer1RADDAM_HE->Fill(amplitudewithPedSubtr);
          h_sigLayer1RADDAM_HE->Fill(double(ieta0), amplitudewithPedSubtr);
          h_sigLayer1RADDAM0_HE->Fill(double(ieta0), 1.);
        }
        if (LLLLLL222222 == 1) {
          //forStudy    h_mapLayer2RADDAM_HE->Fill(fabs(double(ieta0)), amplitudewithPedSubtr); h_mapLayer2RADDAM0_HE->Fill(fabs(double(ieta0)), 1.); h_A_Layer2RADDAM_HE->Fill(amplitudewithPedSubtr);
          h_sigLayer2RADDAM_HE->Fill(double(ieta0), amplitudewithPedSubtr);
          h_sigLayer2RADDAM0_HE->Fill(double(ieta0), 1.);
        }

        //
        if (mdepth == 3 && fabs(ieta0) == 16) {
          h_mapDepth3RADDAM16_HE->Fill(amplitudewithPedSubtr);
          // forStudy     h_mapDepth3RADDAM160_HE->Fill(1.);
        }
        //
      }  //amplitude > 60.
    }    // END RADDAM

    ///////////////////////////////    for HE All
    if (mdepth == 1)
      h_mapDepth1_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HE->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 2 )
     //
}  // fillDigiAmplitude
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeQIE11(QIE11DataFrame qie11df) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  DetId detid = qie11df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE QIE11DigiCollection
  nTS = qie11df.samples();
  /////////////////////////////////////////////////////////////////
  if (mdepth == 0 || sub > 4)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 3)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 7)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 8)
    return;
  if (mdepth > 3 && flagupgradeqie1011_ == 9)
    return;

  // for some CMSSW versions and GT this line uncommented, can help to run job
  //if(mdepth ==4  && sub==1  && (ieta == -16 || ieta == 15)   ) return;// HB depth4 eta=-16, 15 since I did:if(ieta > 0) ieta -= 1;
  /////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////
  const HcalGain* gain = conditions->getGain(hcaldetid);
  //    const HcalGainWidth* gainWidth = conditions->getGainWidth(hcaldetid);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(hcaldetid);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(hcaldetid);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(hcaldetid);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(hcaldetid);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(hcaldetid);
  const HcalPedestal* pedestal00 = conditions->getPedestal(hcaldetid);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(qie11df, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double difpedestal0 = 0.;
  double difpedestal1 = 0.;
  double difpedestal2 = 0.;
  double difpedestal3 = 0.;

  double amplitude = 0.;
  double amplitude0 = 0.;
  double absamplitude = 0.;
  double tocampl = 0.;

  double amplitude345 = 0.;
  double ampl = 0.;
  double ampl3ts = 0.;
  double amplmaxts = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  int TSsize = 10;  // sub= 1 HB
  if (sub == 2)
    TSsize = 8;  // sub = 2 HE

  if (nTS != TSsize)
    errorBtype = 1.;
  TSsize = nTS;  //nTS = qie11df.samples();
  ///////   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // double ADC_ped = 0.;
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double tocdefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;

    ampldefault0 = adc2fC_QIE11_shunt6[qie11df[ii].adc()];  // massive !!!!!!    (use for local runs as default shunt6)
    if (flaguseshunt_ == 1)
      ampldefault0 = adc2fC_QIE11_shunt1[qie11df[ii].adc()];  // massive !!!!!!
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];  //adcfC
    ampldefault2 = qie11df[ii].adc();   //ADCcounts

    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }  // !!!!!!
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    tocdefault = ampldefault;

    int capid = (qie11df[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    double pedestalINI = pedestal00->getValue(capid);
    tocdefault -= pedestal;  // pedestal subtraction
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    tool[ii] = ampldefault;
    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal0 = pedestal - pedestalINI;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal1 = pedestal - pedestalINI;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal2 = pedestal - pedestalINI;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
      difpedestal3 = pedestal - pedestalINI;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    tocampl += tocdefault;             //

    if (ii == 1 || ii == 2 || ii == 3 || ii == 4 || ii == 5 || ii == 6 || ii == 7 || ii == 8)
      amplitude345 += ampldefault;

    if (flagcpuoptimization_ == 0) {
    }  //flagcpuoptimization

    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;

  }  //for 1
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  amplitude0 = amplitude;

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < 10) {
    ampl = tool[ts_with_max_signal];
    ampl3ts = tool[ts_with_max_signal];
    amplmaxts = tool[ts_with_max_signal];
  }
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < 10) {
    ampl += tool[ts_with_max_signal - 1];
    ampl3ts += tool[ts_with_max_signal - 1];
  }
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < 10) {
    ampl += tool[ts_with_max_signal + 1];
    ampl3ts += tool[ts_with_max_signal + 1];
  }
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < 10) {
    ampl += tool[ts_with_max_signal + 2];
  }
  // HE charge correction for SiPMs:
  if (flagsipmcorrection_ != 0) {
    if (sub == 2) {
      double xa = amplitude / 40.;
      double xb = ampl / 40.;
      double xc = amplitude345 / 40.;
      double xd = ampl3ts / 40.;
      double xe = amplmaxts / 40.;
      double txa = tocampl / 40.;
      // ADDI case:
      if (((ieta == -16 || ieta == 15) && mdepth == 4) ||
          ((ieta == -17 || ieta == 16) && (mdepth == 2 || mdepth == 3)) ||
          ((ieta == -18 || ieta == 17) && mdepth == 5)) {
        double c0 = 1.000000;
        double b1 = 2.59096e-05;
        double a2 = 4.60721e-11;
        double corrforxa = a2 * xa * xa + b1 * xa + c0;
        double corrforxb = a2 * xb * xb + b1 * xb + c0;
        double corrforxc = a2 * xc * xc + b1 * xc + c0;
        double corrforxd = a2 * xd * xd + b1 * xd + c0;
        double corrforxe = a2 * xe * xe + b1 * xe + c0;
        double corrfortxa = a2 * txa * txa + b1 * txa + c0;
        h_corrforxaADDI_HE->Fill(amplitude, corrforxa);
        h_corrforxaADDI0_HE->Fill(amplitude, 1.);
        amplitude *= corrforxa;
        ampl *= corrforxb;
        amplitude345 *= corrforxc;
        ampl3ts *= corrforxd;
        amplmaxts *= corrforxe;
        tocampl *= corrfortxa;
      }
      // MAIN case:
      else {
        double c0 = 1.000000;
        double b1 = 2.71238e-05;
        double a2 = 1.32877e-10;
        double corrforxa = a2 * xa * xa + b1 * xa + c0;
        double corrforxb = a2 * xb * xb + b1 * xb + c0;
        double corrforxc = a2 * xc * xc + b1 * xc + c0;
        double corrforxd = a2 * xd * xd + b1 * xd + c0;
        double corrforxe = a2 * xe * xe + b1 * xe + c0;
        double corrfortxa = a2 * txa * txa + b1 * txa + c0;
        h_corrforxaMAIN_HE->Fill(amplitude, corrforxa);
        h_corrforxaMAIN0_HE->Fill(amplitude, 1.);
        amplitude *= corrforxa;
        ampl *= corrforxb;
        amplitude345 *= corrforxc;
        ampl3ts *= corrforxd;
        amplmaxts *= corrforxe;
        tocampl *= corrfortxa;
      }
    }  // sub == 2   HE charge correction end
  }    //flagsipmcorrection_
  ///////   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      !!!!!!!!!!!!!!!!!!

  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71  HBHE
  tocamplchannel[sub - 1][mdepth - 1][ieta + 41][iphi] += tocampl;      // 0-neta ; 0-71  HBHE

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude0 > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude0;  // average_TS +1
  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = (qie11df[ii]).capid();
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
  }  //for 2

  if (!anycapid)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude0 > 0 && rmsamp > 0) || (amplitude0 < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude0);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9
  // CapIdErrors end  /////////////////////////////////////////////////////////

  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_forCapIdErrors_HE3->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 1 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HB1->Fill(amplitude, 1.);
    if (sub == 1 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HB2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HE1->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HE2->Fill(amplitude, 1.);
    if (sub == 2 && mdepth == 3)
      h_Amplitude_notCapIdErrors_HE3->Fill(amplitude, 1.);
  }

  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<10; ii++) {
    double ampldefault = tool[ii];
    ///
    if (sub == 1) {
      if (amplitude0 > 120) {
        h_shape_Ahigh_HB0->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB0->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB0->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB0->Fill(float(ii), 1.);
      }  //HB0
      ///
      if (pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        h_shape_Ahigh_HB1->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB1->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB1->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB1->Fill(float(ii), 1.);
      }  //HB1
      if (error == 0) {
        h_shape_Ahigh_HB2->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB2->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB2->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB2->Fill(float(ii), 1.);
      }  //HB2
      ///
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        h_shape_Ahigh_HB3->Fill(float(ii), ampldefault);
        h_shape0_Ahigh_HB3->Fill(float(ii), 1.);
      } else {
        h_shape_Alow_HB3->Fill(float(ii), ampldefault);
        h_shape0_Alow_HB3->Fill(float(ii), 1.);
      }  //HB3

    }  // sub   HB

  }  //for 3 over TSs

  if (sub == 1) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude0 < ADCAmplHBMin_ || amplitude0 > ADCAmplHBMax_ || rmsamplitude < rmsHBMin_ ||
        rmsamplitude > rmsHBMax_ || pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ ||
        pedestal2 < pedestalHBMax_ || pedestal3 < pedestalHBMax_ || pedestalw0 < pedestalwHBMax_ ||
        pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ || pedestalw3 < pedestalwHBMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HB->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HB->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HB->Fill(float(ii), 1.);
      }
    }
  }  // sub   HB

  // HE starts:
  if (sub == 2) {
    // shape bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude0 < ADCAmplHEMin_ || amplitude0 > ADCAmplHEMax_ || rmsamplitude < rmsHEMin_ ||
        rmsamplitude > rmsHEMax_ || pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ ||
        pedestal2 < pedestalHEMax_ || pedestal3 < pedestalHEMax_ || pedestalw0 < pedestalwHEMax_ ||
        pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ || pedestalw3 < pedestalwHEMax_) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HE->Fill(float(ii), 1.);
      }
    }
    // shape good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HE->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HE->Fill(float(ii), 1.);
      }  // ii
    }    // else for good channels
  }      // sub   HE
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  //    sumEstimator0[sub-1][mdepth-1][ieta+41][iphi] += pedestalw0;//Sig_Pedestals
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HB
  if (sub == 1) {
    if (studyPedestalCorrelations_) {
      //   //   //   //   //   //   //   //   //  HB       PedestalCorrelations :
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HB->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl_HB->Fill(mypedestal, amplitude);
      h_pedwvsampl_HB->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HB->Fill(mypedestal, 1.);
      h_pedwvsampl0_HB->Fill(mypedestalw, 1.);

      h2_amplvsped_HB->Fill(amplitude, mypedestal);
      h2_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped_HB->Fill(amplitude, mypedestal);
      h_amplvspedw_HB->Fill(amplitude, mypedestalw);
      h_amplvsped0_HB->Fill(amplitude, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HB->Fill(pedestal0, 1.);
      h_pedestal1_HB->Fill(pedestal1, 1.);
      h_pedestal2_HB->Fill(pedestal2, 1.);
      h_pedestal3_HB->Fill(pedestal3, 1.);
      h_pedestalaver4_HB->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HB->Fill(pedestalaver9, 1.);
      h_pedestalw0_HB->Fill(pedestalw0, 1.);
      h_pedestalw1_HB->Fill(pedestalw1, 1.);
      h_pedestalw2_HB->Fill(pedestalw2, 1.);
      h_pedestalw3_HB->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HB->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HB->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HB->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HB->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HB->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HB->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HB->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HB->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HB->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HB->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHBMax_ || pedestalw1 < pedestalwHBMax_ || pedestalw2 < pedestalwHBMax_ ||
          pedestalw3 < pedestalwHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth4pedestalw_HB->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHBMax_ || pedestal1 < pedestalHBMax_ || pedestal2 < pedestalHBMax_ ||
          pedestal3 < pedestalHBMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HB->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HB->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HB->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HB->Fill(respcorr->getValue(), 1.);
      h_timecorr_HB->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HB->Fill(lutcorr->getValue(), 1.);
      h_difpedestal0_HB->Fill(difpedestal0, 1.);
      h_difpedestal1_HB->Fill(difpedestal1, 1.);
      h_difpedestal2_HB->Fill(difpedestal2, 1.);
      h_difpedestal3_HB->Fill(difpedestal3, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HB       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345Zoom_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345Zoom1_HB->Fill(amplitude345, 1.);
      h_ADCAmpl345_HB->Fill(amplitude345, 1.);
      if (error == 0) {
        h_ADCAmpl_HBCapIdNoError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdNoError->Fill(amplitude345, 1.);
      }
      if (error == 1) {
        h_ADCAmpl_HBCapIdError->Fill(amplitude, 1.);
        h_ADCAmpl345_HBCapIdError->Fill(amplitude345, 1.);
      }
      h_ADCAmplZoom_HB->Fill(amplitude, 1.);
      h_ADCAmplZoom1_HB->Fill(amplitude, 1.);
      h_ADCAmpl_HB->Fill(amplitude, 1.);

      h_AmplitudeHBrest->Fill(amplitude, 1.);
      h_AmplitudeHBrest1->Fill(amplitude, 1.);
      h_AmplitudeHBrest6->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHBMin_ || amplitude > ADCAmplHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >400.) averSIGNALoccupancy_HB += 1.;
      if (amplitude < 35.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HB->Fill(double(ieta), double(iphi), ampl);
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HB->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HB->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HB->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HB->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HB->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHBMin_ || aveamplitude1 > TSmeanHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HB->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HB->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHBMin_ || ts_with_max_signal > TSpeakHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HB->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HB->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHBMin_ || rmsamplitude > rmsHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HB->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HB->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HB->Fill(ratio, 1.);
      if (ratio < ratioHBMin_ || ratio > ratioHBMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HB->Fill(double(ieta), double(iphi), 1.);
        // //
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HB->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HB      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HB->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HB All
    if (mdepth == 1)
      h_mapDepth1_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HB->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HB->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 1 )

  // HE   QIE11
  if (sub == 2) {
    //   //   //   //   //   //   //   //   //  HE   QIE11    PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      //	double mypedestal  = pedestalaver9;
      //	double mypedestalw = pedestalwaver9;
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HE->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl_HE->Fill(mypedestal, amplitude);
      h_pedwvsampl_HE->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HE->Fill(mypedestal, 1.);
      h_pedwvsampl0_HE->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE   QIE11    Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HE->Fill(pedestal0, 1.);
      h_pedestal1_HE->Fill(pedestal1, 1.);
      h_pedestal2_HE->Fill(pedestal2, 1.);
      h_pedestal3_HE->Fill(pedestal3, 1.);
      h_pedestalaver4_HE->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HE->Fill(pedestalaver9, 1.);
      h_pedestalw0_HE->Fill(pedestalw0, 1.);
      h_pedestalw1_HE->Fill(pedestalw1, 1.);
      h_pedestalw2_HE->Fill(pedestalw2, 1.);
      h_pedestalw3_HE->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HE->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HE->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 3) {
        h_mapDepth3Ped0_HE->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth3Ped1_HE->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth3Ped2_HE->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth3Ped3_HE->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth3Pedw0_HE->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth3Pedw1_HE->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth3Pedw2_HE->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth3Pedw3_HE->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHEMax_ || pedestalw1 < pedestalwHEMax_ || pedestalw2 < pedestalwHEMax_ ||
          pedestalw3 < pedestalwHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7pedestalw_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHEMax_ || pedestal1 < pedestalHEMax_ || pedestal2 < pedestalHEMax_ ||
          pedestal3 < pedestalHEMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6pedestal_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7pedestal_HE->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HE->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HE->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HE->Fill(respcorr->getValue(), 1.);
      h_timecorr_HE->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HE->Fill(lutcorr->getValue(), 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HE  QIE11     ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl345_HE->Fill(amplitude345, 1.);
      h_ADCAmpl_HE->Fill(amplitude, 1.);
      //	if( ieta <0) h_ADCAmpl_HEM->Fill(amplitude,1.);
      //	if( ieta >0) h_ADCAmpl_HEP->Fill(amplitude,1.);
      h_ADCAmplrest_HE->Fill(amplitude, 1.);
      h_ADCAmplrest1_HE->Fill(amplitude, 1.);
      h_ADCAmplrest6_HE->Fill(amplitude, 1.);

      if (amplitude < ADCAmplHEMin_ || amplitude > ADCAmplHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7ADCAmpl225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude > 700.) averSIGNALoccupancy_HE += 1.;
      if (amplitude < 500.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7ADCAmpl225Copy_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if

      h_ADCAmplZoom1_HE->Fill(amplitude, 1.);   // for amplitude allTS
      h_ADCAmpl345Zoom1_HE->Fill(ampl3ts, 1.);  // for ampl3ts 3TSs
      h_ADCAmpl345Zoom_HE->Fill(ampl, 1.);      // for ampl 4TSs

      if (amplitude > 110 && amplitude < 150) {
        h_mapADCAmplfirstpeak_HE->Fill(double(ieta), double(iphi), amplitude);
        h_mapADCAmplfirstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (amplitude > 150 && amplitude < 190) {
        h_mapADCAmplsecondpeak_HE->Fill(double(ieta), double(iphi), amplitude);
        h_mapADCAmplsecondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }

      if (ampl3ts > 70 && ampl3ts < 110) {
        h_mapADCAmpl11firstpeak_HE->Fill(double(ieta), double(iphi), ampl3ts);
        h_mapADCAmpl11firstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (ampl3ts > 110 && ampl3ts < 150) {
        h_mapADCAmpl11secondpeak_HE->Fill(double(ieta), double(iphi), ampl3ts);
        h_mapADCAmpl11secondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }
      if (ampl > 87 && ampl < 127) {
        h_mapADCAmpl12firstpeak_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapADCAmpl12firstpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      } else if (ampl > 127 && ampl < 167) {
        h_mapADCAmpl12secondpeak_HE->Fill(double(ieta), double(iphi), ampl);
        h_mapADCAmpl12secondpeak0_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values of every channel:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 5)
        h_mapDepth5ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 6)
        h_mapDepth6ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 7)
        h_mapDepth7ADCAmpl_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 5)
        h_mapDepth5ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 6)
        h_mapDepth6ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 7)
        h_mapDepth7ADCAmpl12_HE->Fill(double(ieta), double(iphi), ampl);
      // for averaged values of SiPM channels only:
      if (mdepth == 1)
        h_mapDepth1ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmplSiPM_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12SiPM_HE->Fill(double(ieta), double(iphi), ampl);
      //
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 21.09.2018 for Pavel Bunin:
      ///////////////////////////////////////////////////////////////////////////////	//AZ: 25.10.2018 for Pavel Bunin: gain stability vs LSs using LED from abort gap
      h_bcnvsamplitude_HE->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HE->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HE->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HE->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    //   //   //   //   //   //   //   //   //  HE  QIE11     TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HE->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHEMin_ || aveamplitude1 > TSmeanHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7TSmeanA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 5)
        h_mapDepth5TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 6)
        h_mapDepth6TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 7)
        h_mapDepth7TSmeanA_HE->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE  QIE11     TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HE->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHEMin_ || ts_with_max_signal > TSpeakHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7TSmaxA225_HE->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 5)
        h_mapDepth5TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 6)
        h_mapDepth6TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 7)
        h_mapDepth7TSmaxA_HE->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_) {
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE   QIE11    RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HE->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHEMin_ || rmsamplitude > rmsHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7Amplitude225_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 5)
        h_mapDepth5Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 6)
        h_mapDepth6Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 7)
        h_mapDepth7Amplitude_HE->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HE  QIE11     Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HE->Fill(ratio, 1.);
      if (ratio < ratioHEMin_ || ratio > ratioHEMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 5)
          h_mapDepth5Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 6)
          h_mapDepth6Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 7)
          h_mapDepth7Ampl047_HE->Fill(double(ieta), double(iphi), 1.);
      }
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 5)
        h_mapDepth5Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 6)
        h_mapDepth6Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 7)
        h_mapDepth7Ampl_HE->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HE   QIE11    DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 5)
        h_mapDepth5AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 6)
        h_mapDepth6AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 7)
        h_mapDepth7AmplE34_HE->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)
    ///////////////////////////////    for HE All QIE11
    if (mdepth == 1)
      h_mapDepth1_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 5)
      h_mapDepth5_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 6)
      h_mapDepth6_HE->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 7)
      h_mapDepth7_HE->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 2 )
     //
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHF(HFDigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // (HFDigiCollection: 4-HF)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;
  int TSsize = 4;
  if ((*digiItr).size() != TSsize)
    errorBtype = 1.;
  TSsize = digiItr->size();
  for (int ii = 0; ii < TSsize; ii++) {
    //  for (int ii=0; ii<digiItr->size(); ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }

    int capid = ((*digiItr)[ii]).capid();
    //      double pedestal = calib.pedestal(capid);
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction

    tool[ii] = ampldefault;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    ///////////////////////////////////

    if (flagcpuoptimization_ == 0) {
    }  //  if(flagcpuoptimization_== 0
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }                                                                     //for 1
  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71 HF

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;

  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;

  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1

  double rmsamp = 0.;
  // and CapIdErrors:
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    //    std::cout << " ii = " << ii  << " capid = " << capid  << " ((lastcapid+1)%4) = " << ((lastcapid+1)%4)  << std::endl;
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2

  if (!anycapid || anyer || !anydv)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9
  if (error == 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HF2->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HF2->Fill(amplitude, 1.);
  }

  if (sub == 4) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_ || rmsamplitude < rmsHFMin_ ||
        rmsamplitude > rmsHFMax_ || pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ ||
        pedestal2 < pedestalHFMax_ || pedestal3 < pedestalHFMax_ || pedestalw0 < pedestalwHFMax_ ||
        pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ || pedestalw3 < pedestalwHFMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HF->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HF->Fill(float(ii), 1.);
      }
    }
  }  // sub   HF
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //    Pedestals
  // HF
  if (sub == 4) {
    //   //   //   //   //   //   //   //   //  HF      PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HF->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl_HF->Fill(mypedestal, amplitude);
      h_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HF->Fill(mypedestal, 1.);
      h_pedwvsampl0_HF->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HF       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HF->Fill(pedestal0, 1.);
      h_pedestal1_HF->Fill(pedestal1, 1.);
      h_pedestal2_HF->Fill(pedestal2, 1.);
      h_pedestal3_HF->Fill(pedestal3, 1.);
      h_pedestalaver4_HF->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HF->Fill(pedestalaver9, 1.);
      h_pedestalw0_HF->Fill(pedestalw0, 1.);
      h_pedestalw1_HF->Fill(pedestalw1, 1.);
      h_pedestalw2_HF->Fill(pedestalw2, 1.);
      h_pedestalw3_HF->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HF->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HF->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }

      if (pedestalw0 < pedestalwHFMax_ || pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ ||
          pedestalw3 < pedestalwHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
      }

      if (pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ || pedestal2 < pedestalHFMax_ ||
          pedestal3 < pedestalHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HF->Fill(double(ieta), double(iphi), 1.);
      }

      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HF->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HF->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HF->Fill(respcorr->getValue(), 1.);
      h_timecorr_HF->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HF->Fill(lutcorr->getValue(), 1.);

    }  //

    //   //   //   //   //   //   //   //   //  HF       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HF->Fill(amplitude, 1.);
      h_ADCAmplrest1_HF->Fill(amplitude, 1.);
      h_ADCAmplrest6_HF->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HF->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >1500.) averSIGNALoccupancy_HF += 1.;
      if (amplitude < 20.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if

      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HF->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HF->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HF->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HF->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    ///////////////////////////////

    //   //   //   //   //   //   //   //   //  HF       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HF->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHFMin_ || aveamplitude1 > TSmeanHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HF->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHFMin_ || ts_with_max_signal > TSpeakHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HF->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHFMin_ || rmsamplitude > rmsHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HF->Fill(ratio, 1.);
      if (ratio < ratioHFMin_ || ratio > ratioHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HF->Fill(double(ieta), double(iphi), ratio);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)

    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HF      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HF All
    if (mdepth == 1)
      h_mapDepth1_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HF->Fill(double(ieta), double(iphi), 1.);

  }  //if ( sub == 4 )

  //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHFQIE10(QIE10DataFrame qie10df) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  DetId detid = qie10df.detid();
  HcalDetId hcaldetid = HcalDetId(detid);
  int ieta = hcaldetid.ieta();
  if (ieta > 0)
    ieta -= 1;
  int iphi = hcaldetid.iphi() - 1;
  int mdepth = hcaldetid.depth();
  int sub = hcaldetid.subdet();  // 1-HB, 2-HE (HFQIE10DigiCollection: 4-HF)
  nTS = qie10df.samples();       //  ----------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  /*
                                  # flag   HBHE8    HBHE11   HF8   HF10  comments:
                                  #  0       +        +       +     +     all
                                  #  1       +        -       +     -     old
                                  #  2       -        +       -     +     new (2018)
                                  #  3       -        +       -     +     new w/o high depthes
                                  #  4       +        -       +     +     2016fall
                                  #  5       +        -       +     +     2016fall w/o high depthes
                                  #  6       +        +       -     +     2017begin
                                  #  7       +        +       -     +     2017begin w/o high depthes in HEonly
                                  #  8       +        +       -     +     2017begin w/o high depthes
                                  #  9       +        +       +     +     all  w/o high depthes
*/
  if (mdepth == 0 || sub != 4)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 3)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 5)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 8)
    return;
  if (mdepth > 2 && flagupgradeqie1011_ == 9)
    return;
  /////////////////////////////////////////////////////////////////
  //    HcalCalibrations calib = conditions->getHcalCalibrations(hcaldetid);
  const HcalPedestal* pedestal00 = conditions->getPedestal(hcaldetid);
  const HcalGain* gain = conditions->getGain(hcaldetid);
  //  const HcalGainWidth* gainWidth = conditions->getGainWidth(hcaldetid);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(hcaldetid);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(hcaldetid);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(hcaldetid);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(hcaldetid);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(hcaldetid);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(qie10df, toolOriginal);
  //    double noiseADC = qie10df[0].adc();
  /////////////////////////////////////////////////////////////////
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;

  int TSsize = 3;
  if (nTS != TSsize)
    errorBtype = 1.;
  TSsize = nTS;  // ------------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC_QIE10[qie10df[ii].adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];  //adcfC
    ampldefault2 = qie10df[ii].adc();   //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }

    int capid = (qie10df[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);

    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction

    tool[ii] = ampldefault;

    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;

    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;          //
    absamplitude += abs(ampldefault);  //
    ///////////////////////////////////
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }                                                                     //for 1
  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71 HF

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);

  // ------------ to get signal in TS: -2 max +1  ------------
  if (ts_with_max_signal > -1 && ts_with_max_signal < TSsize)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < TSsize)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < TSsize)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < TSsize)
    ampl += tool[ts_with_max_signal - 1];

  double ratio = 0.;
  //    if(amplallTS != 0.) ratio = ampl/amplallTS;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.02)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1

  double rmsamp = 0.;
  int error = 0;
  bool anycapid = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = (qie10df[ii]).capid();
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
  }  //for 2

  if (!anycapid)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9

  // CapIdErrors end  /////////////////////////////////////////////////////////
  // AZ 1.10.2015:
  if (error == 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_forCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_forCapIdErrors_HF2->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 4 && mdepth == 1)
      h_Amplitude_notCapIdErrors_HF1->Fill(amplitude, 1.);
    if (sub == 4 && mdepth == 2)
      h_Amplitude_notCapIdErrors_HF2->Fill(amplitude, 1.);
  }

  if (sub == 4) {
    // bad_channels with C,A,W,P,pW,
    if (error == 1 || amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_ || rmsamplitude < rmsHFMin_ ||
        rmsamplitude > rmsHFMax_ || pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ ||
        pedestal2 < pedestalHFMax_ || pedestal3 < pedestalHFMax_ || pedestalw0 < pedestalwHFMax_ ||
        pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ || pedestalw3 < pedestalwHFMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HF->Fill(float(ii), 1.);
      }
    }
    // good_channels with C,A,W,P,pW,
    else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HF->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HF->Fill(float(ii), 1.);
      }
    }
  }  // sub   HFQIE10
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //    Pedestals
  // HFQIE10
  if (sub == 4) {
    //   //   //   //   //   //   //   //   //  HFQIE10      PedestalCorrelations :
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HF->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl_HF->Fill(mypedestal, amplitude);
      h_pedwvsampl_HF->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HF->Fill(mypedestal, 1.);
      h_pedwvsampl0_HF->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HFQIE10       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HF->Fill(pedestal0, 1.);
      h_pedestal1_HF->Fill(pedestal1, 1.);
      h_pedestal2_HF->Fill(pedestal2, 1.);
      h_pedestal3_HF->Fill(pedestal3, 1.);
      h_pedestalaver4_HF->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HF->Fill(pedestalaver9, 1.);
      h_pedestalw0_HF->Fill(pedestalw0, 1.);
      h_pedestalw1_HF->Fill(pedestalw1, 1.);
      h_pedestalw2_HF->Fill(pedestalw2, 1.);
      h_pedestalw3_HF->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HF->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HF->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 1) {
        h_mapDepth1Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth1Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth1Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth1Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth1Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth1Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth1Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth1Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (mdepth == 2) {
        h_mapDepth2Ped0_HF->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth2Ped1_HF->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth2Ped2_HF->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth2Ped3_HF->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth2Pedw0_HF->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth2Pedw1_HF->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth2Pedw2_HF->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth2Pedw3_HF->Fill(double(ieta), double(iphi), pedestalw3);
      }

      if (pedestalw0 < pedestalwHFMax_ || pedestalw1 < pedestalwHFMax_ || pedestalw2 < pedestalwHFMax_ ||
          pedestalw3 < pedestalwHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestalw_HF->Fill(double(ieta), double(iphi), 1.);
      }

      if (pedestal0 < pedestalHFMax_ || pedestal1 < pedestalHFMax_ || pedestal2 < pedestalHFMax_ ||
          pedestal3 < pedestalHFMax_) {
        if (mdepth == 1)
          h_mapDepth1pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3pedestal_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4pedestal_HF->Fill(double(ieta), double(iphi), 1.);
      }

      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HF->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HF->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HF->Fill(respcorr->getValue(), 1.);
      h_timecorr_HF->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HF->Fill(lutcorr->getValue(), 1.);

    }  //

    //   //   //   //   //   //   //   //   //  HFQIE10       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HF->Fill(amplitude, 1.);
      h_ADCAmplrest1_HF->Fill(amplitude, 1.);
      h_ADCAmplrest6_HF->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HF->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHFMin_ || amplitude > ADCAmplHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >1500.) averSIGNALoccupancy_HF += 1.;
      if (amplitude < 20.) {
        if (mdepth == 1)
          h_mapDepth1ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if

      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 1)
        h_mapDepth1ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 2)
        h_mapDepth2ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 3)
        h_mapDepth3ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HF->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HF->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HF->Fill(float(bcn), 1.);
      h_orbitNumvsamplitude_HF->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HF->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    //   //   //   //   //   //   //   //   //  HFQIE10       TSmean:
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HF->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHFMin_ || aveamplitude1 > TSmeanHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 2)
        h_mapDepth2TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 3)
        h_mapDepth3TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HF->Fill(double(ieta), double(iphi), aveamplitude1);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       TSmax:
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HF->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHFMin_ || ts_with_max_signal > TSpeakHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 2)
        h_mapDepth2TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 3)
        h_mapDepth3TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HF->Fill(double(ieta), double(iphi), float(ts_with_max_signal));

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       RMS:
    if (studyRMSshapeHist_) {
      h_Amplitude_HF->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHFMin_ || rmsamplitude > rmsHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HF->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:

      if (mdepth == 1)
        h_mapDepth1Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 2)
        h_mapDepth2Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 3)
        h_mapDepth3Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (mdepth == 4)
        h_mapDepth4Amplitude_HF->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10       Ratio:
    if (studyRatioShapeHist_) {
      h_Ampl_HF->Fill(ratio, 1.);
      if (ratio < ratioHFMin_ || ratio > ratioHFMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 1)
          h_mapDepth1Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 2)
          h_mapDepth2Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 3)
          h_mapDepth3Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
        if (mdepth == 4)
          h_mapDepth4Ampl047_HF->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      // for averaged values:
      if (mdepth == 1)
        h_mapDepth1Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 2)
        h_mapDepth2Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 3)
        h_mapDepth3Ampl_HF->Fill(double(ieta), double(iphi), ratio);
      if (mdepth == 4)
        h_mapDepth4Ampl_HF->Fill(double(ieta), double(iphi), ratio);

      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)

    ///////////////////////////////
    //   //   //   //   //   //   //   //   //  HFQIE10      DiffAmplitude:
    if (studyDiffAmplHist_) {
      if (mdepth == 1)
        h_mapDepth1AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 2)
        h_mapDepth2AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 3)
        h_mapDepth3AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4AmplE34_HF->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)

    ///////////////////////////////    for HFQIE10 All
    if (mdepth == 1)
      h_mapDepth1_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 2)
      h_mapDepth2_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 3)
      h_mapDepth3_HF->Fill(double(ieta), double(iphi), 1.);
    if (mdepth == 4)
      h_mapDepth4_HF->Fill(double(ieta), double(iphi), 1.);

  }  //if ( sub == 4 )

  //
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CMTRawAnalyzer::fillDigiAmplitudeHO(HODigiCollection::const_iterator& digiItr) {
  CaloSamples toolOriginal;  // TS
  double tool[100];
  HcalDetId cell(digiItr->id());
  int mdepth = cell.depth();
  int iphi = cell.iphi() - 1;  // 0-71
  int ieta = cell.ieta();
  if (ieta > 0)
    ieta -= 1;              //-41 +41
  int sub = cell.subdet();  // (HODigiCollection: 3-HO)
  const HcalPedestal* pedestal00 = conditions->getPedestal(cell);
  const HcalGain* gain = conditions->getGain(cell);
  const HcalRespCorr* respcorr = conditions->getHcalRespCorr(cell);
  const HcalTimeCorr* timecorr = conditions->getHcalTimeCorr(cell);
  const HcalLUTCorr* lutcorr = conditions->getHcalLUTCorr(cell);
  const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
  const HcalPedestalWidth* pedw = conditions->getPedestalWidth(cell);
  HcalCoderDb coder(*channelCoder, *shape);
  if (useADCfC_)
    coder.adc2fC(*digiItr, toolOriginal);
  double pedestalaver9 = 0.;
  double pedestalaver4 = 0.;
  double pedestal0 = 0.;
  double pedestal1 = 0.;
  double pedestal2 = 0.;
  double pedestal3 = 0.;
  double pedestalwaver9 = 0.;
  double pedestalwaver4 = 0.;
  double pedestalw0 = 0.;
  double pedestalw1 = 0.;
  double pedestalw2 = 0.;
  double pedestalw3 = 0.;
  double amplitude = 0.;
  double absamplitude = 0.;
  double ampl = 0.;
  double timew = 0.;
  double timeww = 0.;
  double max_signal = -100.;
  int ts_with_max_signal = -100;
  int c0 = 0;
  int c1 = 0;
  int c2 = 0;
  int c3 = 0;
  int c4 = 0;
  double errorBtype = 0.;
  int TSsize = 10;
  if ((*digiItr).size() != TSsize)
    errorBtype = 1.;
  TSsize = digiItr->size();
  for (int ii = 0; ii < TSsize; ii++) {
    double ampldefault = 0.;
    double ampldefault0 = 0.;
    double ampldefault1 = 0.;
    double ampldefault2 = 0.;
    ampldefault0 = adc2fC[digiItr->sample(ii).adc()];  // massive
    if (useADCfC_)
      ampldefault1 = toolOriginal[ii];    //adcfC
    ampldefault2 = (*digiItr)[ii].adc();  //ADCcounts
    if (useADCmassive_) {
      ampldefault = ampldefault0;
    }
    if (useADCfC_) {
      ampldefault = ampldefault1;
    }
    if (useADCcounts_) {
      ampldefault = ampldefault2;
    }
    int capid = ((*digiItr)[ii]).capid();
    double pedestal = pedestal00->getValue(capid);
    double pedestalw = pedw->getSigma(capid, capid);
    if (usePedestalSubtraction_)
      ampldefault -= pedestal;  // pedestal subtraction
    tool[ii] = ampldefault;
    pedestalaver9 += pedestal;
    pedestalwaver9 += pedestalw * pedestalw;
    if (capid == 0 && c0 == 0) {
      c0++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal0 = pedestal;
      pedestalw0 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 1 && c1 == 0) {
      c1++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal1 = pedestal;
      pedestalw1 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 2 && c2 == 0) {
      c2++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal2 = pedestal;
      pedestalw2 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }
    if (capid == 3 && c3 == 0) {
      c3++;
      c4++;
      pedestalaver4 += pedestal;
      pedestal3 = pedestal;
      pedestalw3 = pedestalw;
      pedestalwaver4 += pedestalw * pedestalw;
    }

    if (max_signal < ampldefault) {
      max_signal = ampldefault;
      ts_with_max_signal = ii;
    }
    amplitude += ampldefault;
    absamplitude += abs(ampldefault);
    ///////////////////////////////////////////
    if (flagcpuoptimization_ == 0) {
    }
    timew += (ii + 1) * abs(ampldefault);
    timeww += (ii + 1) * ampldefault;
  }                                                                     //for 1
  amplitudechannel[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;  // 0-neta ; 0-71  HO

  pedestalaver9 /= TSsize;
  pedestalaver4 /= c4;
  pedestalwaver9 = sqrt(pedestalwaver9 / TSsize);
  pedestalwaver4 = sqrt(pedestalwaver4 / c4);
  if (ts_with_max_signal > -1 && ts_with_max_signal < 10)
    ampl = tool[ts_with_max_signal];
  if (ts_with_max_signal + 2 > -1 && ts_with_max_signal + 2 < 10)
    ampl += tool[ts_with_max_signal + 2];
  if (ts_with_max_signal + 1 > -1 && ts_with_max_signal + 1 < 10)
    ampl += tool[ts_with_max_signal + 1];
  if (ts_with_max_signal - 1 > -1 && ts_with_max_signal - 1 < 10)
    ampl += tool[ts_with_max_signal - 1];
  double ratio = 0.;
  if (amplitude != 0.)
    ratio = ampl / amplitude;
  if (ratio < 0. || ratio > 1.04)
    ratio = 0.;
  double aveamplitude = 0.;
  double aveamplitudew = 0.;
  if (absamplitude > 0 && timew > 0)
    aveamplitude = timew / absamplitude;  // average_TS +1
  if (amplitude > 0 && timeww > 0)
    aveamplitudew = timeww / amplitude;  // average_TS +1
  double rmsamp = 0.;
  int error = 0;
  bool anycapid = true;
  bool anyer = false;
  bool anydv = true;
  int lastcapid = 0;
  int capid = 0;
  for (int ii = 0; ii < TSsize; ii++) {
    double aaaaaa = (ii + 1) - aveamplitudew;
    double aaaaaa2 = aaaaaa * aaaaaa;
    double ampldefault = tool[ii];
    rmsamp += (aaaaaa2 * ampldefault);  // fC
    capid = ((*digiItr)[ii]).capid();
    bool er = (*digiItr)[ii].er();  // error
    bool dv = (*digiItr)[ii].dv();  // valid data
    if (ii != 0 && ((lastcapid + 1) % 4) != capid) {
      anycapid = false;
    }
    lastcapid = capid;
    if (er) {
      anyer = true;
    }
    if (!dv) {
      anydv = false;
    }
  }  //for 2

  if (!anycapid || anyer || !anydv)
    error = 1;
  double rmsamplitude = 0.;
  if ((amplitude > 0 && rmsamp > 0) || (amplitude < 0 && rmsamp < 0))
    rmsamplitude = sqrt(rmsamp / amplitude);
  double aveamplitude1 = aveamplitude - 1;  // means iTS=0-9, so bad is iTS=0 and 9
  if (error == 1) {
    if (sub == 3 && mdepth == 4)
      h_Amplitude_forCapIdErrors_HO4->Fill(amplitude, 1.);
  }
  if (error != 1) {
    if (sub == 3 && mdepth == 4)
      h_Amplitude_notCapIdErrors_HO4->Fill(amplitude, 1.);
  }

  if (sub == 3) {
    if (error == 1 || amplitude < ADCAmplHOMin_ || amplitude > ADCAmplHOMax_ || rmsamplitude < rmsHOMin_ ||
        rmsamplitude > rmsHOMax_ || pedestal0 < pedestalHOMax_ || pedestal1 < pedestalHOMax_ ||
        pedestal2 < pedestalHOMax_ || pedestal3 < pedestalHOMax_ || pedestalw0 < pedestalwHOMax_ ||
        pedestalw1 < pedestalwHOMax_ || pedestalw2 < pedestalwHOMax_ || pedestalw3 < pedestalwHOMax_

    ) {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_bad_channels_HO->Fill(float(ii), ampldefault);
        h_shape0_bad_channels_HO->Fill(float(ii), 1.);
      }
    } else {
      for (int ii = 0; ii < TSsize; ii++) {
        double ampldefault = tool[ii];
        h_shape_good_channels_HO->Fill(float(ii), ampldefault);
        h_shape0_good_channels_HO->Fill(float(ii), 1.);
      }
    }
  }  // sub   HO
  ///////////////////////////////////////Digis : over all digiHits
  sum0Estimator[sub - 1][mdepth - 1][ieta + 41][iphi] += 1.;
  //      for Error B-type
  sumEstimator6[sub - 1][mdepth - 1][ieta + 41][iphi] += errorBtype;
  sumEstimator0[sub - 1][mdepth - 1][ieta + 41][iphi] += pedestal0;  //Pedestals
  // HO
  if (sub == 3) {
    if (studyPedestalCorrelations_) {
      double mypedestal = pedestal0;
      double mypedestalw = pedestalw0;
      h2_pedvsampl_HO->Fill(mypedestal, amplitude);
      h2_pedwvsampl_HO->Fill(mypedestalw, amplitude);
      h_pedvsampl_HO->Fill(mypedestal, amplitude);
      h_pedwvsampl_HO->Fill(mypedestalw, amplitude);
      h_pedvsampl0_HO->Fill(mypedestal, 1.);
      h_pedwvsampl0_HO->Fill(mypedestalw, 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HO       Pedestals:
    if (studyPedestalsHist_) {
      h_pedestal0_HO->Fill(pedestal0, 1.);
      h_pedestal1_HO->Fill(pedestal1, 1.);
      h_pedestal2_HO->Fill(pedestal2, 1.);
      h_pedestal3_HO->Fill(pedestal3, 1.);
      h_pedestalaver4_HO->Fill(pedestalaver4, 1.);
      h_pedestalaver9_HO->Fill(pedestalaver9, 1.);
      h_pedestalw0_HO->Fill(pedestalw0, 1.);
      h_pedestalw1_HO->Fill(pedestalw1, 1.);
      h_pedestalw2_HO->Fill(pedestalw2, 1.);
      h_pedestalw3_HO->Fill(pedestalw3, 1.);
      h_pedestalwaver4_HO->Fill(pedestalwaver4, 1.);
      h_pedestalwaver9_HO->Fill(pedestalwaver9, 1.);
      // for averaged values:
      if (mdepth == 4) {
        h_mapDepth4Ped0_HO->Fill(double(ieta), double(iphi), pedestal0);
        h_mapDepth4Ped1_HO->Fill(double(ieta), double(iphi), pedestal1);
        h_mapDepth4Ped2_HO->Fill(double(ieta), double(iphi), pedestal2);
        h_mapDepth4Ped3_HO->Fill(double(ieta), double(iphi), pedestal3);
        h_mapDepth4Pedw0_HO->Fill(double(ieta), double(iphi), pedestalw0);
        h_mapDepth4Pedw1_HO->Fill(double(ieta), double(iphi), pedestalw1);
        h_mapDepth4Pedw2_HO->Fill(double(ieta), double(iphi), pedestalw2);
        h_mapDepth4Pedw3_HO->Fill(double(ieta), double(iphi), pedestalw3);
      }
      if (pedestalw0 < pedestalwHOMax_ || pedestalw1 < pedestalwHOMax_ || pedestalw2 < pedestalwHOMax_ ||
          pedestalw3 < pedestalwHOMax_) {
        if (mdepth == 4)
          h_mapDepth4pedestalw_HO->Fill(double(ieta), double(iphi), 1.);
      }
      if (pedestal0 < pedestalHOMax_ || pedestal1 < pedestalHOMax_ || pedestal2 < pedestalHOMax_ ||
          pedestal3 < pedestalHOMax_) {
        if (mdepth == 4)
          h_mapDepth4pedestal_HO->Fill(double(ieta), double(iphi), 1.);
      }
      for (int ii = 0; ii < TSsize; ii++) {
        h_pedestal00_HO->Fill(pedestal00->getValue(ii), 1.);
        h_gain_HO->Fill(gain->getValue(ii), 1.);
      }
      h_respcorr_HO->Fill(respcorr->getValue(), 1.);
      h_timecorr_HO->Fill(timecorr->getValue(), 1.);
      h_lutcorr_HO->Fill(lutcorr->getValue(), 1.);
    }  //
    //   //   //   //   //   //   //   //   //  HO       ADCAmpl:
    if (studyADCAmplHist_) {
      h_ADCAmpl_HO->Fill(amplitude, 1.);
      h_ADCAmplrest1_HO->Fill(amplitude, 1.);
      h_ADCAmplrest6_HO->Fill(amplitude, 1.);

      h_ADCAmplZoom1_HO->Fill(amplitude, 1.);
      h_ADCAmpl_HO_copy->Fill(amplitude, 1.);
      if (amplitude < ADCAmplHOMin_ || amplitude > ADCAmplHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 5)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      //	if(amplitude >2000.) averSIGNALoccupancy_HO += 1.;

      if (amplitude < 100.) {
        if (mdepth == 4)
          h_mapDepth4ADCAmpl225Copy_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 4)
        h_mapDepth4ADCAmpl_HO->Fill(double(ieta), double(iphi), amplitude);
      if (mdepth == 4)
        h_mapDepth4ADCAmpl12_HO->Fill(double(ieta), double(iphi), ampl);

      h_bcnvsamplitude_HO->Fill(float(bcn), amplitude);
      h_bcnvsamplitude0_HO->Fill(float(bcn), 1.);

      h_orbitNumvsamplitude_HO->Fill(float(orbitNum), amplitude);
      h_orbitNumvsamplitude0_HO->Fill(float(orbitNum), 1.);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator1[sub - 1][mdepth - 1][ieta + 41][iphi] += amplitude;
    }  //if(studyADCAmplHist_
    if (studyTSmeanShapeHist_) {
      h_TSmeanA_HO->Fill(aveamplitude1, 1.);
      if (aveamplitude1 < TSmeanHOMin_ || aveamplitude1 > TSmeanHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 4)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4TSmeanA225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      if (mdepth == 4)
        h_mapDepth4TSmeanA_HO->Fill(double(ieta), double(iphi), aveamplitude1);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator2[sub - 1][mdepth - 1][ieta + 41][iphi] += aveamplitude1;
    }  //if(studyTSmeanShapeHist_
    if (studyTSmaxShapeHist_) {
      h_TSmaxA_HO->Fill(float(ts_with_max_signal), 1.);
      if (ts_with_max_signal < TSpeakHOMin_ || ts_with_max_signal > TSpeakHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 3)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4TSmaxA225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      // for averaged values:
      if (mdepth == 4)
        h_mapDepth4TSmaxA_HO->Fill(double(ieta), double(iphi), float(ts_with_max_signal));
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator3[sub - 1][mdepth - 1][ieta + 41][iphi] += float(ts_with_max_signal);
    }  //if(studyTSmaxShapeHist_
    if (studyRMSshapeHist_) {
      h_Amplitude_HO->Fill(rmsamplitude, 1.);
      if (rmsamplitude < rmsHOMin_ || rmsamplitude > rmsHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 2)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4Amplitude225_HO->Fill(double(ieta), double(iphi), 1.);
      }  // if
      if (mdepth == 4)
        h_mapDepth4Amplitude_HO->Fill(double(ieta), double(iphi), rmsamplitude);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator4[sub - 1][mdepth - 1][ieta + 41][iphi] += rmsamplitude;
    }  //if(studyRMSshapeHist_)
    if (studyRatioShapeHist_) {
      h_Ampl_HO->Fill(ratio, 1.);
      if (ratio < ratioHOMin_ || ratio > ratioHOMax_) {
        if (studyRunDependenceHist_ && flagtodefinebadchannel_ == 1)
          ++badchannels[sub - 1][mdepth - 1][ieta + 41][iphi];  // 0-neta ; 0-71
        if (mdepth == 4)
          h_mapDepth4Ampl047_HO->Fill(double(ieta), double(iphi), 1.);
      }  //if(ratio
      if (mdepth == 4)
        h_mapDepth4Ampl_HO->Fill(double(ieta), double(iphi), ratio);
      if (amplitude > forallestimators_amplitude_bigger_)
        sumEstimator5[sub - 1][mdepth - 1][ieta + 41][iphi] += ratio;
    }  //if(studyRatioShapeHist_)
    if (studyDiffAmplHist_) {
      if (mdepth == 4)
        h_mapDepth4AmplE34_HO->Fill(double(ieta), double(iphi), amplitude);
    }  // if(studyDiffAmplHist_)
    if (mdepth == 4)
      h_mapDepth4_HO->Fill(double(ieta), double(iphi), 1.);
  }  //if ( sub == 3 )
}
int CMTRawAnalyzer::getRBX(int& kdet, int& keta, int& kphi) {
  int cal_RBX = 0;
  if (kdet == 1 || kdet == 2) {
    if (kphi == 71)
      cal_RBX = 0;
    else
      cal_RBX = (kphi + 1) / 4;
    cal_RBX = cal_RBX + 18 * (keta + 1) / 2;
  }
  if (kdet == 4) {
    cal_RBX = (int)(kphi / 18) + 1;
  }
  if (kdet == 3) {
    if (keta == -2) {
      if (kphi == 71)
        cal_RBX = 0;
      else
        cal_RBX = kphi / 12 + 1;
    }
    if (keta == -1) {
      if (kphi == 71)
        cal_RBX = 6;
      else
        cal_RBX = kphi / 12 + 1 + 6;
    }
    if (keta == 0) {
      if (kphi == 71)
        cal_RBX = 12;
      else
        cal_RBX = kphi / 6 + 1 + 12;
    }
    if (keta == 1) {
      if (kphi == 71)
        cal_RBX = 24;
      else
        cal_RBX = kphi / 12 + 1 + 24;
    }
    if (keta == 2) {
      if (kphi == 71)
        cal_RBX = 30;
      else
        cal_RBX = kphi / 12 + 1 + 30;
    }
  }
  return cal_RBX;
}
void CMTRawAnalyzer::beginRun(const edm::Run& r, const edm::EventSetup& iSetup) {}
void CMTRawAnalyzer::endRun(const edm::Run& r, const edm::EventSetup& iSetup) {
  if (flagfitshunt1pedorledlowintensity_ != 0) {
  }  // if flag...
  if (usecontinuousnumbering_) {
    lscounterM1 = lscounter - 1;
  } else {
    lscounterM1 = ls0;
  }
  if (ls0 != -1)
    h_nevents_per_eachRealLS->Fill(float(lscounterM1), float(nevcounter));  //
  h_nevents_per_LS->Fill(float(nevcounter));
  h_nevents_per_LSzoom->Fill(float(nevcounter));
  nevcounter0 = nevcounter;
  if (nevcounter0 != 0) {
    for (int k0 = 0; k0 < nsub; k0++) {
      for (int k1 = 0; k1 < ndepth; k1++) {
        for (int k2 = 0; k2 < neta; k2++) {
          for (int k3 = 0; k3 < nphi; k3++) {
            int ieta = k2 - 41;
            if (sumEstimator0[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator0[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator0[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator0[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }
              if (k0 == 0) {
                if (k1 + 1 == 1) {
                  h_sumPedestalLS1->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS2->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumPedestalLS3->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS4->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumPedestalLS5->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumPedestalLS6->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumPedestalLS7->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumPedestalLS8->Fill(bbbc / bbb1);
                  h_2DsumPedestalLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumPedestalLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumPedestalperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0PedestalperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator0[k0][k1][k2][k3] != 0.
            if (sumEstimator1[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator1[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator1[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator1[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }
              if (lscounterM1 >= lsmin_ && lscounterM1 < lsmax_) {
                int kkkk2 = (k2 - 1) / 4;
                if (k2 == 0)
                  kkkk2 = 1.;
                else
                  kkkk2 += 2;              //kkkk2= 1-22
                int kkkk3 = (k3) / 4 + 1;  //kkkk3= 1-18
                int ietaphi = 0;
                ietaphi = ((kkkk2)-1) * znphi + (kkkk3);
                double bbb3 = 0.;
                if (bbb1 != 0.)
                  bbb3 = bbbc / bbb1;
                if (k0 == 0) {
                  h_2DsumADCAmplEtaPhiLs0->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HB
                  h_2DsumADCAmplEtaPhiLs00->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HB
                }
                if (k0 == 1) {
                  h_2DsumADCAmplEtaPhiLs1->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HE
                  h_2DsumADCAmplEtaPhiLs10->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HE
                }
                if (k0 == 2) {
                  h_2DsumADCAmplEtaPhiLs2->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HO
                  h_2DsumADCAmplEtaPhiLs20->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HO
                }
                if (k0 == 3) {
                  h_2DsumADCAmplEtaPhiLs3->Fill(float(lscounterM1), float(ietaphi), bbbc);   //HF
                  h_2DsumADCAmplEtaPhiLs30->Fill(float(lscounterM1), float(ietaphi), bbb1);  //HF
                }

                h_sumADCAmplEtaPhiLs->Fill(bbb3);
                h_sumADCAmplEtaPhiLs_bbbc->Fill(bbbc);
                h_sumADCAmplEtaPhiLs_bbb1->Fill(bbb1);
                h_sumADCAmplEtaPhiLs_lscounterM1orbitNum->Fill(float(lscounterM1), float(orbitNum));
                h_sumADCAmplEtaPhiLs_orbitNum->Fill(float(orbitNum), 1.);
                h_sumADCAmplEtaPhiLs_lscounterM1->Fill(float(lscounterM1), 1.);
                h_sumADCAmplEtaPhiLs_ietaphi->Fill(float(ietaphi));
              }  // lscounterM1 >= lsmin_ && lscounterM1 < lsmax_
              if (k0 == 0) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS1copy1->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy2->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy3->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy4->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1copy5->Fill(bbbc / bbb1);
                  h_sumADCAmplLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                    h_2DsumADCAmplLS1->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth1_)
                    h_2DsumADCAmplLS1_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth1_)
                    h_sumCutADCAmplperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                    h_2DsumADCAmplLS2->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HBdepth2_)
                    h_2DsumADCAmplLS2_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth2_)
                    h_sumCutADCAmplperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS2->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                    h_sumCutADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth3HBu->Fill(float(lscounterM1), bbb1);

                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth3_)
                    h_2DsumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth3HBu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==3)
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                    h_sumCutADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth4HBu->Fill(float(lscounterM1), bbb1);

                  if (bbbc / bbb1 > lsdep_estimator1_HBdepth4_)
                    h_2DsumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HBu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
              }
              if (k0 == 1) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                    h_2DsumADCAmplLS3->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth1_)
                    h_2DsumADCAmplLS3_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth1_)
                    h_sumCutADCAmplperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                    h_2DsumADCAmplLS4->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth2_)
                    h_2DsumADCAmplLS4_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth2_)
                    h_sumCutADCAmplperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumADCAmplLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                    h_2DsumADCAmplLS5->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HEdepth3_)
                    h_2DsumADCAmplLS5_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth3_)
                    h_sumCutADCAmplperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS5->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                    h_sumCutADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth4HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth4_)
                    h_2DsumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
                if (k1 + 1 == 5) {
                  h_sumADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                    h_sumCutADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth5HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth5_)
                    h_2DsumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth5HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==5)

                if (k1 + 1 == 6) {
                  h_sumADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                    h_sumCutADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth6HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth6_)
                    h_2DsumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth6HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==6)
                if (k1 + 1 == 7) {
                  h_sumADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                    h_sumCutADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLSdepth7HEu->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HEdepth7_)
                    h_2DsumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth7HEu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==7)

              }  // end HE

              if (k0 == 3) {
                if (k1 + 1 == 1) {
                  h_sumADCAmplLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                    h_2DsumADCAmplLS6->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth1_)
                    h_2DsumADCAmplLS6_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth1_)
                    h_sumCutADCAmplperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumADCAmplLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                    h_2DsumADCAmplLS7->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HFdepth2_)
                    h_2DsumADCAmplLS7_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth2_)
                    h_sumCutADCAmplperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS7->Fill(float(lscounterM1), bbb1);
                }

                if (k1 + 1 == 3) {
                  h_sumADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                    h_sumCutADCAmplperLS6u->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS6u->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth3_)
                    h_2DsumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth3HFu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==3)
                if (k1 + 1 == 4) {
                  h_sumADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                    h_sumCutADCAmplperLS7u->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS7u->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HFdepth4_)
                    h_2DsumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLSdepth4HFu->Fill(double(ieta), double(k3), bbb1);
                }  //if(k1+1  ==4)
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumADCAmplLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                    h_2DsumADCAmplLS8->Fill(double(ieta), double(k3), bbbc);
                  if (bbbc / bbb1 > 2. * lsdep_estimator1_HOdepth4_)
                    h_2DsumADCAmplLS8_LSselected->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumADCAmplLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator1_HOdepth4_)
                    h_sumCutADCAmplperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0ADCAmplperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator1[k0][k1][k2][k3] != 0.
            if (sumEstimator2[k0][k1][k2][k3] != 0.) {
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator2[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator2[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator2[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                    h_2DsumTSmeanALS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth1_)
                    h_sumCutTSmeanAperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator2_HBdepth1_)
                    h_sumTSmeanAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                    h_2DsumTSmeanALS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HBdepth2_)
                    h_sumCutTSmeanAperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              if (k0 == 1) {
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                    h_2DsumTSmeanALS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth1_)
                    h_sumCutTSmeanAperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                    h_2DsumTSmeanALS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth2_)
                    h_sumCutTSmeanAperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumTSmeanALS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                    h_2DsumTSmeanALS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HEdepth3_)
                    h_sumCutTSmeanAperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmeanALS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                    h_2DsumTSmeanALS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth1_)
                    h_sumCutTSmeanAperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmeanALS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                    h_2DsumTSmeanALS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HFdepth2_)
                    h_sumCutTSmeanAperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumTSmeanALS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                    h_2DsumTSmeanALS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmeanALS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator2_HOdepth4_)
                    h_sumCutTSmeanAperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmeanAperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator2[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator3 Tx
            if (sumEstimator3[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator3[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator3[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator3[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                    h_2DsumTSmaxALS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth1_)
                    h_sumCutTSmaxAperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator3_HBdepth1_)
                    h_sumTSmaxAperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                    h_2DsumTSmaxALS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HBdepth2_)
                    h_sumCutTSmaxAperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                    h_2DsumTSmaxALS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth1_)
                    h_sumCutTSmaxAperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                    h_2DsumTSmaxALS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth2_)
                    h_sumCutTSmaxAperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumTSmaxALS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                    h_2DsumTSmaxALS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HEdepth3_)
                    h_sumCutTSmaxAperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumTSmaxALS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                    h_2DsumTSmaxALS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth1_)
                    h_sumCutTSmaxAperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumTSmaxALS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                    h_2DsumTSmaxALS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HFdepth2_)
                    h_sumCutTSmaxAperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumTSmaxALS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                    h_2DsumTSmaxALS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumTSmaxALS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator3_HOdepth4_)
                    h_sumCutTSmaxAperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0TSmaxAperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator3[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator4 W
            if (sumEstimator4[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator4[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator4[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator4[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                    h_2DsumAmplitudeLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth1_)
                    h_sumCutAmplitudeperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator4_HBdepth1_)
                    h_sumAmplitudeperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                    h_2DsumAmplitudeLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HBdepth2_)
                    h_sumCutAmplitudeperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                    h_2DsumAmplitudeLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth1_)
                    h_sumCutAmplitudeperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                    h_2DsumAmplitudeLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth2_)
                    h_sumCutAmplitudeperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumAmplitudeLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                    h_2DsumAmplitudeLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HEdepth3_)
                    h_sumCutAmplitudeperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplitudeLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                    h_2DsumAmplitudeLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth1_)
                    h_sumCutAmplitudeperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplitudeLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                    h_2DsumAmplitudeLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HFdepth2_)
                    h_sumCutAmplitudeperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumAmplitudeLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                    h_2DsumAmplitudeLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplitudeLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator4_HOdepth4_)
                    h_sumCutAmplitudeperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplitudeperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator4[k0][k1][k2][k3] != 0.

            // ------------------------------------------------------------------------------------------------------------------------sumEstimator5 R
            if (sumEstimator5[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator5[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator5[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator5[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS1->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                    h_2DsumAmplLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS1->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth1_)
                    h_sumCutAmplperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS1->Fill(float(lscounterM1), bbb1);
                  if (bbbc / bbb1 > 2. * lsdep_estimator5_HBdepth1_)
                    h_sumAmplperLS1_LSselected->Fill(float(lscounterM1), bbbc);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS2->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                    h_2DsumAmplLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS2->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HBdepth2_)
                    h_sumCutAmplperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS3->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                    h_2DsumAmplLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS3->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth1_)
                    h_sumCutAmplperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS4->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                    h_2DsumAmplLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS4->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth2_)
                    h_sumCutAmplperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumAmplLS5->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                    h_2DsumAmplLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS5->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HEdepth3_)
                    h_sumCutAmplperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumAmplLS6->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                    h_2DsumAmplLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS6->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth1_)
                    h_sumCutAmplperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumAmplLS7->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                    h_2DsumAmplLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS7->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HFdepth2_)
                    h_sumCutAmplperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth1
                if (k1 + 1 == 4) {
                  h_sumAmplLS8->Fill(bbbc / bbb1);
                  if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                    h_2DsumAmplLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumAmplLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumAmplperLS8->Fill(float(lscounterM1), bbbc);
                  if (bbbc / bbb1 > lsdep_estimator5_HOdepth4_)
                    h_sumCutAmplperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0AmplperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
            }  //if(sumEstimator5[k0][k1][k2][k3] != 0.
            // ------------------------------------------------------------------------------------------------------------------------sumEstimator6 (Error-B)
            if (sumEstimator6[k0][k1][k2][k3] != 0.) {
              // fill histoes:
              double bbbc = 0.;
              if (flagestimatornormalization_ == 0)
                bbbc = sumEstimator6[k0][k1][k2][k3] / nevcounter0;
              if (flagestimatornormalization_ == 1)
                bbbc = sumEstimator6[k0][k1][k2][k3] / sum0Estimator[k0][k1][k2][k3];
              double bbb1 = 1.;
              if (flagestimatornormalization_ == 2) {
                bbbc = sumEstimator6[k0][k1][k2][k3];
                bbb1 = sum0Estimator[k0][k1][k2][k3];
              }

              // HB:
              if (k0 == 0) {
                // HBdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS1->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS1->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS1->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS1->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS1->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS2->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS2->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS2->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS2->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS2->Fill(float(lscounterM1), bbb1);
                }
              }
              // HE:
              if (k0 == 1) {
                // HEdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS3->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS3->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS3->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS3->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS3->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS4->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS4->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS4->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS4->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS4->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 3) {
                  h_sumErrorBLS5->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS5->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS5->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS5->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS5->Fill(float(lscounterM1), bbb1);
                }
              }
              // HF:
              if (k0 == 3) {
                // HFdepth1
                if (k1 + 1 == 1) {
                  h_sumErrorBLS6->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS6->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS6->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS6->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS6->Fill(float(lscounterM1), bbb1);
                }
                if (k1 + 1 == 2) {
                  h_sumErrorBLS7->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS7->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS7->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS7->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS7->Fill(float(lscounterM1), bbb1);
                }
              }
              // HO:
              if (k0 == 2) {
                // HOdepth4
                if (k1 + 1 == 4) {
                  h_sumErrorBLS8->Fill(bbbc / bbb1);
                  h_2DsumErrorBLS8->Fill(double(ieta), double(k3), bbbc);
                  h_2D0sumErrorBLS8->Fill(double(ieta), double(k3), bbb1);
                  h_sumErrorBperLS8->Fill(float(lscounterM1), bbbc);
                  h_sum0ErrorBperLS8->Fill(float(lscounterM1), bbb1);
                }
              }
              ///
            }  //if(sumEstimator6[k0][k1][k2][k3] != 0.

            ///
            ///
          }  //for
        }    //for
      }      //for
    }        //for

    //------------------------------------------------------   averSIGNAL
    averSIGNALoccupancy_HB /= float(nevcounter0);
    h_averSIGNALoccupancy_HB->Fill(float(lscounterM1), averSIGNALoccupancy_HB);
    averSIGNALoccupancy_HE /= float(nevcounter0);
    h_averSIGNALoccupancy_HE->Fill(float(lscounterM1), averSIGNALoccupancy_HE);
    averSIGNALoccupancy_HF /= float(nevcounter0);
    h_averSIGNALoccupancy_HF->Fill(float(lscounterM1), averSIGNALoccupancy_HF);
    averSIGNALoccupancy_HO /= float(nevcounter0);
    h_averSIGNALoccupancy_HO->Fill(float(lscounterM1), averSIGNALoccupancy_HO);

    averSIGNALoccupancy_HB = 0.;
    averSIGNALoccupancy_HE = 0.;
    averSIGNALoccupancy_HF = 0.;
    averSIGNALoccupancy_HO = 0.;

    //------------------------------------------------------
    averSIGNALsumamplitude_HB /= float(nevcounter0);
    h_averSIGNALsumamplitude_HB->Fill(float(lscounterM1), averSIGNALsumamplitude_HB);
    averSIGNALsumamplitude_HE /= float(nevcounter0);
    h_averSIGNALsumamplitude_HE->Fill(float(lscounterM1), averSIGNALsumamplitude_HE);
    averSIGNALsumamplitude_HF /= float(nevcounter0);
    h_averSIGNALsumamplitude_HF->Fill(float(lscounterM1), averSIGNALsumamplitude_HF);
    averSIGNALsumamplitude_HO /= float(nevcounter0);
    h_averSIGNALsumamplitude_HO->Fill(float(lscounterM1), averSIGNALsumamplitude_HO);

    averSIGNALsumamplitude_HB = 0.;
    averSIGNALsumamplitude_HE = 0.;
    averSIGNALsumamplitude_HF = 0.;
    averSIGNALsumamplitude_HO = 0.;

    //------------------------------------------------------   averNOSIGNAL
    averNOSIGNALoccupancy_HB /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HB->Fill(float(lscounterM1), averNOSIGNALoccupancy_HB);
    averNOSIGNALoccupancy_HE /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HE->Fill(float(lscounterM1), averNOSIGNALoccupancy_HE);
    averNOSIGNALoccupancy_HF /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HF->Fill(float(lscounterM1), averNOSIGNALoccupancy_HF);
    averNOSIGNALoccupancy_HO /= float(nevcounter0);
    h_averNOSIGNALoccupancy_HO->Fill(float(lscounterM1), averNOSIGNALoccupancy_HO);

    averNOSIGNALoccupancy_HB = 0.;
    averNOSIGNALoccupancy_HE = 0.;
    averNOSIGNALoccupancy_HF = 0.;
    averNOSIGNALoccupancy_HO = 0.;

    //------------------------------------------------------
    averNOSIGNALsumamplitude_HB /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HB->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HB);
    averNOSIGNALsumamplitude_HE /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HE->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HE);
    averNOSIGNALsumamplitude_HF /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HF->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HF);
    averNOSIGNALsumamplitude_HO /= float(nevcounter0);
    h_averNOSIGNALsumamplitude_HO->Fill(float(lscounterM1), averNOSIGNALsumamplitude_HO);

    averNOSIGNALsumamplitude_HB = 0.;
    averNOSIGNALsumamplitude_HE = 0.;
    averNOSIGNALsumamplitude_HF = 0.;
    averNOSIGNALsumamplitude_HO = 0.;

    h_maxxSUMAmpl_HB->Fill(float(lscounterM1), maxxSUM1);
    h_maxxSUMAmpl_HE->Fill(float(lscounterM1), maxxSUM2);
    h_maxxSUMAmpl_HO->Fill(float(lscounterM1), maxxSUM3);
    h_maxxSUMAmpl_HF->Fill(float(lscounterM1), maxxSUM4);
    maxxSUM1 = 0.;
    maxxSUM2 = 0.;
    maxxSUM3 = 0.;
    maxxSUM4 = 0.;
    //------------------------------------------------------
    h_maxxOCCUP_HB->Fill(float(lscounterM1), maxxOCCUP1);
    h_maxxOCCUP_HE->Fill(float(lscounterM1), maxxOCCUP2);
    h_maxxOCCUP_HO->Fill(float(lscounterM1), maxxOCCUP3);
    h_maxxOCCUP_HF->Fill(float(lscounterM1), maxxOCCUP4);
    maxxOCCUP1 = 0.;
    maxxOCCUP2 = 0.;
    maxxOCCUP3 = 0.;
    maxxOCCUP4 = 0.;

  }  //if( nevcounter0 != 0 )
     /////////////////////////////// -------------------------------------------------------------------

  std::cout << " ==== Edn of run " << std::endl;
}
/////////////////////////////// -------------------------------------------------------------------
/////////////////////////////// -------------------------------------------------------------------
/////////////////////////////// -------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(CMTRawAnalyzer);
