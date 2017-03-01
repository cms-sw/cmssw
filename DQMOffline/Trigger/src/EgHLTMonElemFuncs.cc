#include "DQMOffline/Trigger/interface/EgHLTMonElemFuncs.h"

#include "DQMOffline/Trigger/interface/EgHLTMonElemMgrEBEE.h"
#include "DQMOffline/Trigger/interface/EgHLTEgCutCodes.h"
#include "DQMOffline/Trigger/interface/EgHLTDQMCut.h"
#include "DQMOffline/Trigger/interface/EgHLTCutMasks.h"


using namespace egHLT;

void MonElemFuncs::initStdEleHists(std::vector<MonElemManagerBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins)
{
  addStdHist<OffEle,float>(histVec,baseName+"_energy",baseName+" reco CaloEnergy;reco CaloEnergy (GeV)",bins.energy, &OffEle::energy);
  addStdHist<OffEle,float>(histVec,baseName+"_et",baseName+" E_{T};E_{T} (GeV)",bins.et,&OffEle::et); 
  addStdHist<OffEle,float>(histVec,baseName+"_etHigh",baseName+" E_{T};E_{T} (GeV)",bins.etHigh,&OffEle::et); 
  addStdHist<OffEle,float>(histVec,baseName+"_etSC",baseName+" E^{SC}_{T};E^{SC}_{T} (GeV)",bins.et,&OffEle::etSC);
  addStdHist<OffEle,float>(histVec,baseName+"_eta",baseName+" #eta;#eta",bins.eta,&OffEle::detEta);		
  addStdHist<OffEle,float>(histVec,baseName+"_phi",baseName+" #phi;#phi (rad)",bins.phi,&OffEle::phi);
  // addStdHist<OffEle,int>(histVec,baseName+"_charge",baseName+" Charge; charge",bins.charge,&OffEle::charge);
  
  addStdHist<OffEle,float>(histVec,baseName+"_hOverE",baseName+" H/E; H/E",bins.hOverE,&OffEle::hOverE);
  //----Morse
  addStdHist<OffEle,float>(histVec,baseName+"_maxr9",baseName+" MAXR9 ; MAXR9",bins.maxr9,&OffEle::r9);
  addStdHist<OffEle,float>(histVec,baseName+"_HLTenergy",baseName+" HLT Energy;HLT Energy (GeV)",bins.HLTenergy,&OffEle::hltEnergy); 
  addStdHist<OffEle,float>(histVec,baseName+"_HLTeta",baseName+" HLT #eta;HLT #eta",bins.HLTeta,&OffEle::hltEta);		
  addStdHist<OffEle,float>(histVec,baseName+"_HLTphi",baseName+" HLT #phi;HLT #phi (rad)",bins.HLTphi,&OffEle::hltPhi);
  //-------
  addStdHist<OffEle,float>(histVec,baseName+"_dPhiIn",baseName+" #Delta #phi_{in}; #Delta #phi_{in}",bins.dPhiIn,&OffEle::dPhiIn);
  addStdHist<OffEle,float>(histVec,baseName+"_dEtaIn",baseName+" #Delta #eta_{in}; #Delta #eta_{in}",bins.dEtaIn,&OffEle::dEtaIn);
  addStdHist<OffEle,float>(histVec,baseName+"_sigmaIEtaIEta",baseName+"#sigma_{i#etai#eta}; #sigma_{i#etai#eta}",bins.sigEtaEta,&OffEle::sigmaIEtaIEta);  
  addStdHist<OffEle,float>(histVec,baseName+"_epIn",baseName+"E/p_{in}; E/p_{in}",bins.eOverP,&OffEle::epIn);
  addStdHist<OffEle,float>(histVec,baseName+"_epOut",baseName+"E/p_{out}; E/p_{out}",bins.eOverP,&OffEle::epOut); 
  addStdHist<OffEle,float>(histVec,baseName+"_invEInvP",baseName+"1/E -1/p; 1/E - 1/p",bins.invEInvP,&OffEle::invEInvP);

  addStdHist<OffEle,float>(histVec,baseName+"_e2x5Over5x5",baseName+"E^{2x5}/E^{5x5}; E^{2x5}/E^{5x5}",bins.e2x5,&OffEle::e2x5MaxOver5x5);
  addStdHist<OffEle,float>(histVec,baseName+"_e1x5Over5x5",baseName+"E^{1x5}/E^{5x5}; E^{1x5}/E^{5x5}",bins.e1x5,&OffEle::e1x5Over5x5);
  //addStdHist<OffEle,float>(histVec,baseName+"_isolEM",baseName+"Isol EM; Isol EM (GeV)",bins.isolEm,&OffEle::isolEm); 
  //addStdHist<OffEle,float>(histVec,baseName+"_isolHad",baseName+"Isol Had; Isol Had (GeV)",bins.isolHad,&OffEle::isolHad);
  //addStdHist<OffEle,float>(histVec,baseName+"_isolPtTrks",baseName+"Isol Pt Trks; Isol Pt Tracks (GeV/c)",bins.isolPtTrks,&OffEle::isolPtTrks); 
  addStdHist<OffEle,float>(histVec,baseName+"_hltIsolTrksEle",baseName+"HLT Ele Isol Trks; HLT Ele Iso Tracks (GeV/c)",bins.isolPtTrks,&OffEle::hltIsolTrksEle);  
  //addStdHist<OffEle,float>(histVec,baseName+"_hltIsolTrksPho",baseName+"HLT Pho Isol Trks; HLT Pho Iso Tracks (GeV/c)",bins.isolPtTrks,&OffEle::hltIsolTrksPho); 
  addStdHist<OffEle,float>(histVec,baseName+"_hltIsolHad",baseName+"HLT Isol Had; HLT Isol Had (GeV)",bins.isolHad,&OffEle::hltIsolHad);
  addStdHist<OffEle,float>(histVec,baseName+"_hltIsolEm",baseName+"HLT Isol Em; HLT Isol Em (GeV)",bins.isolEm,&OffEle::hltIsolEm);
  addStdHist<OffEle,float>(histVec,baseName+"_DeltaE",baseName+"HLT Energy - reco SC Energy;HLT Energy - reco SC Energy (GeV)",bins.deltaE,&OffEle::DeltaE);

  histVec.push_back(new MonElemManager2D<OffEle,float,float>(iBooker, baseName+"_etaVsPhi",
							     baseName+" #eta vs #phi;#eta;#phi (rad)",
							     bins.etaVsPhi.nrX,bins.etaVsPhi.xMin,bins.etaVsPhi.xMax,
							     bins.etaVsPhi.nrY,bins.etaVsPhi.yMin,bins.etaVsPhi.yMax,
							     &OffEle::detEta,&OffEle::phi));
  histVec.push_back(new MonElemManager2D<OffEle,float,float>(iBooker, baseName+"_HLTetaVsHLTphi",
							     baseName+" HLT #eta vs HLT #phi;HLT #eta;HLT #phi (rad)",
							     bins.etaVsPhi.nrX,bins.etaVsPhi.xMin,bins.etaVsPhi.xMax,
							     bins.etaVsPhi.nrY,bins.etaVsPhi.yMin,bins.etaVsPhi.yMax,
							     &OffEle::hltEta,&OffEle::hltPhi));
}

void MonElemFuncs::initStdPhoHists(std::vector<MonElemManagerBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins)
{
  addStdHist<OffPho,float>(histVec,baseName+"_energy",baseName+" reco Energy;reco Energy (GeV)",bins.energy, &OffPho::energy);
  addStdHist<OffPho,float>(histVec,baseName+"_et",baseName+" E_{T};E_{T} (GeV)",bins.et,&OffPho::et); 
  addStdHist<OffPho,float>(histVec,baseName+"_etHigh",baseName+" E_{T};E_{T} (GeV)",bins.etHigh,&OffPho::et);
  addStdHist<OffPho,float>(histVec,baseName+"_etSC",baseName+" E^{SC}_{T};E^{SC}_{T} (GeV)",bins.et,&OffPho::etSC);
  addStdHist<OffPho,float>(histVec,baseName+"_eta",baseName+" #eta;#eta",bins.eta,&OffPho::detEta);		
  addStdHist<OffPho,float>(histVec,baseName+"_phi",baseName+" #phi;#phi (rad)",bins.phi,&OffPho::phi);
  
  addStdHist<OffPho,float>(histVec,baseName+"_hOverE",baseName+" H/E; H/E",bins.hOverE,&OffPho::hOverE);
  //----Morse
  //addStdHist<OffPho,float>(histVec,baseName+"_r9",baseName+" R9 ; R9",bins.r9,&OffPho::r9);
  //addStdHist<OffPho,float>(histVec,baseName+"_minr9",baseName+" MINR9 ; MINR9",bins.minr9,&OffPho::r9);
  addStdHist<OffPho,float>(histVec,baseName+"_maxr9",baseName+" MAXR9 ; MAXR9",bins.maxr9,&OffPho::r9);
  addStdHist<OffPho,float>(histVec,baseName+"_HLTenergy",baseName+" HLT Energy;HLT Energy (GeV)",bins.HLTenergy,&OffPho::hltEnergy); 
  addStdHist<OffPho,float>(histVec,baseName+"_HLTeta",baseName+" HLT #eta;HLT #eta",bins.HLTeta,&OffPho::hltEta);		
  addStdHist<OffPho,float>(histVec,baseName+"_HLTphi",baseName+" HLT #phi;HLT #phi (rad)",bins.HLTphi,&OffPho::hltPhi);
  //-------
  addStdHist<OffPho,float>(histVec,baseName+"_sigmaIEtaIEta",baseName+"#sigma_{i#etai#eta}; #sigma_{i#etai#eta}",bins.sigEtaEta,&OffPho::sigmaIEtaIEta);  
  addStdHist<OffPho,float>(histVec,baseName+"_e2x5Over5x5",baseName+"E^{2x5}/E^{5x5}; E^{2x5}/E^{5x5}",bins.e2x5,&OffPho::e2x5MaxOver5x5);
  addStdHist<OffPho,float>(histVec,baseName+"_e1x5Over5x5",baseName+"E^{1x5}/E^{5x5}; E^{1x5}/E^{5x5}",bins.e1x5,&OffPho::e1x5Over5x5);
  addStdHist<OffPho,float>(histVec,baseName+"_isolEM",baseName+"Isol EM; Isol EM (GeV)",bins.isolEm,&OffPho::isolEm); 
  addStdHist<OffPho,float>(histVec,baseName+"_isolHad",baseName+"Isol Had; Isol Had (GeV)",bins.isolHad,&OffPho::isolHad);
  addStdHist<OffPho,float>(histVec,baseName+"_isolPtTrks",baseName+"Isol Pt Trks; Isol Pt Tracks (GeV/c)",bins.isolPtTrks,&OffPho::isolPtTrks);  
  addStdHist<OffPho,int>(histVec,baseName+"_isolNrTrks",baseName+"Isol Nr Trks; Isol Nr Tracks",bins.isolNrTrks,&OffPho::isolNrTrks); 
  //addStdHist<OffPho,float>(histVec,baseName+"_hltIsolTrks",baseName+"HLT Isol Trks; HLT Iso Tracks (GeV/c)",bins.isolPtTrks,&OffPho::hltIsolTrks); 
  //addStdHist<OffPho,float>(histVec,baseName+"_hltIsolHad",baseName+"HLT Isol Had; HLT Isol Had (GeV)",bins.isolPtTrks,&OffPho::hltIsolHad);
  addStdHist<OffPho,float>(histVec,baseName+"_DeltaE",baseName+"HLT Energy - reco SC Energy;HLT Energy - reco SC Energy (GeV)",bins.deltaE,&OffPho::DeltaE);
  
  histVec.push_back(new MonElemManager2D<OffPho,float,float>(iBooker, baseName+"_etaVsPhi",
							     baseName+" #eta vs #phi;#eta;#phi (rad)",
							     bins.etaVsPhi.nrX,bins.etaVsPhi.xMin,bins.etaVsPhi.xMax,
							     bins.etaVsPhi.nrY,bins.etaVsPhi.yMin,bins.etaVsPhi.yMax,
							     &OffPho::detEta,&OffPho::phi));
  histVec.push_back(new MonElemManager2D<OffPho,float,float>(iBooker, baseName+"_HLTetaVsHLTphi",
							     baseName+" HLT #eta vs HLT #phi;HLT #eta;HLT #phi (rad)",
							     bins.etaVsPhi.nrX,bins.etaVsPhi.xMin,bins.etaVsPhi.xMax,
							     bins.etaVsPhi.nrY,bins.etaVsPhi.yMin,bins.etaVsPhi.yMax,
							     &OffPho::hltEta,&OffPho::hltPhi));
}
 
void MonElemFuncs::initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData::Data1D& bins,float (OffEle::*vsVarFunc)()const,const CutMasks& masks)
{
  initStdEffHists(histVec,filterName,baseName,bins.nr,bins.min,bins.max,vsVarFunc,masks);
}
  
void MonElemFuncs::initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData::Data1D& bins,float (OffPho::*vsVarFunc)()const,const CutMasks& masks)
{
  initStdEffHists(histVec,filterName,baseName,bins.nr,bins.min,bins.max,vsVarFunc,masks);
}

void MonElemFuncs::initStdEffHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffEle::*vsVarFunc)()const,const CutMasks& masks)
{
  //some convience typedefs, I hate typedefs but atleast here where they are defined is obvious
  typedef EgHLTDQMVarCut<OffEle> VarCut;
  typedef MonElemWithCutEBEE<OffEle,float> MonElemFloat;
  int stdCutCode = masks.stdEle;

  //first do the zero and all cuts histograms
  histVec.push_back(new MonElemFloat(iBooker, baseName+"_noCuts",baseName+" NoCuts",nrBins,xMin,xMax,vsVarFunc));
  histVec.push_back(new MonElemFloat(iBooker, baseName+"_allCuts",baseName+" All Cuts",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(stdCutCode,&OffEle::cutCode))); 
		       
  //now for the n-1
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_dEtaIn",baseName+" N1 #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::DETAIN&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_dPhiIn",baseName+" N1 #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::DPHIIN&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_sigmaIEtaIEta",baseName+" N1 #sigma_{i#etai#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::SIGMAIETAIETA&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_hOverE",baseName+" N1 H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::HADEM&stdCutCode,&OffEle::cutCode)));
  /* histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolEm",baseName+" N1 Isol Em",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(~EgCutCodes::ISOLEM&stdCutCode,&OffEle::cutCode)));
     histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolHad",baseName+" N1 Isol Had",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(~EgCutCodes::ISOLHAD&stdCutCode,&OffEle::cutCode)));
     histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolPtTrks",baseName+" N1 Isol Tracks",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(~EgCutCodes::ISOLPTTRKS&stdCutCode,&OffEle::cutCode)));*/
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_hltIsolEm",baseName+" N1 HLT Isol Em",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::HLTISOLEM&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_hltIsolHad",baseName+" N1 HLT Isol Had",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::HLTISOLHAD&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_hltIsolTrksEle",baseName+" N1 HLT Isol Tracks Ele ",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::HLTISOLTRKSELE&stdCutCode,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_dEtaIn",baseName+" Single #Delta#eta_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::DETAIN,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_dPhiIn",baseName+" Single #Delta#phi_{in}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::DPHIIN,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_sigmaIEtaIEta",baseName+" Single #sigma_{i#etai#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::SIGMAIETAIETA,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hOverE",baseName+" Single H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::HADEM,&OffEle::cutCode)));
  /* histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolEm",baseName+" Single Isol Em",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(EgCutCodes::ISOLEM,&OffEle::cutCode)));
     histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolHad",baseName+" Single Isol Had",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(EgCutCodes::ISOLHAD,&OffEle::cutCode)));
     histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolPtTrks",baseName+" Single Isol Tracks",nrBins,xMin,xMax,vsVarFunc,
     new VarCut(EgCutCodes::ISOLPTTRKS,&OffEle::cutCode)));*/
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolEm",baseName+" Single HLT Isol Em",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::HLTISOLEM,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolHad",baseName+" Single HLT Isol Had",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::HLTISOLHAD,&OffEle::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolTrksEle",baseName+" Single HLT Isol Tracks Ele ",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::HLTISOLTRKSELE,&OffEle::cutCode))); 
  /*histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolTrksPho",baseName+" Single HLT Isol Tracks Pho ",nrBins,xMin,xMax,vsVarFunc,
    new VarCut(EgCutCodes::HLTISOLTRKSPHO,&OffEle::cutCode)));*/
  
}

void MonElemFuncs::initStdEffHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,int nrBins,double xMin,double xMax,float (OffPho::*vsVarFunc)()const,const CutMasks& masks)
{
  //some convenience typedefs, I hate typedefs but atleast here where they are defined is obvious
  typedef EgHLTDQMVarCut<OffPho> VarCut;
  typedef MonElemWithCutEBEE<OffPho,float> MonElemFloat;
  int stdCutCode = masks.stdPho;

  histVec.push_back(new MonElemFloat(iBooker,baseName+"_noCuts",baseName+" NoCuts",nrBins,xMin,xMax,vsVarFunc));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_allCuts",baseName+" All Cuts",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(stdCutCode,&OffPho::cutCode))); 

  
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_sigmaIEtaIEta",baseName+" N1 #sigma_{i#etai#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::SIGMAIETAIETA&stdCutCode,&OffPho::cutCode)));
  //-----Morse------
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_hOverE",baseName+" N1 H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::HADEM&stdCutCode,&OffPho::cutCode)));//---BUG FIX!!--
  /*histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_minr9",baseName+" N1 MINR9",nrBins,xMin,xMax,vsVarFunc,
    new VarCut(~EgCutCodes::MINR9&stdCutCode,&OffPho::cutCode)));
    histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_maxr9",baseName+" N1 MAXR9",nrBins,xMin,xMax,vsVarFunc,
    new VarCut(~EgCutCodes::MAXR9&stdCutCode,&OffPho::cutCode)));*/
  //----------------
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolEm",baseName+" N1 Isol Em",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLEM&stdCutCode,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolHad",baseName+" N1 Isol Had",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLHAD&stdCutCode,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolPtTrks",baseName+" N1 Pt Isol Tracks",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLPTTRKS&stdCutCode,&OffPho::cutCode))); 
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_n1_isolNrTrks",baseName+" N1 Nr Isol Tracks",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLNRTRKS&stdCutCode,&OffPho::cutCode)));

  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hOverE",baseName+" Single H/E",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::HADEM,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_sigmaIEtaIEta",baseName+" Single #sigma_{i#etai#eta}",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(EgCutCodes::SIGMAIETAIETA,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolEm",baseName+" Single Isol Em",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLEM,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolHad",baseName+" Single Isol Had",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLHAD,&OffPho::cutCode)));
  histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_isolPtTrks",baseName+" Single Pt Isol Tracks",nrBins,xMin,xMax,vsVarFunc,
				     new VarCut(~EgCutCodes::ISOLPTTRKS,&OffPho::cutCode))); 
  /*histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolHad",baseName+" Single HLT Isol Had",nrBins,xMin,xMax,vsVarFunc,
    new VarCut(EgCutCodes::HLTISOLHAD,&OffPho::cutCode)));
    histVec.push_back(new MonElemFloat(iBooker,baseName+"_single_hltIsolTrksPho",baseName+" Single HLT Isol Tracks Pho ",nrBins,xMin,xMax,vsVarFunc,
    new VarCut(EgCutCodes::HLTISOLTRKSPHO,&OffPho::cutCode)));*/

  
}


//we own the passed in cut, so we give it to the first mon element and then clone it after that
//only currently used for trigger tag and probe
void MonElemFuncs::initStdEleCutHists(std::vector<MonElemWithCutBase<OffEle>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins,EgHLTDQMCut<OffEle>* cut)
{
  histVec.push_back(new MonElemWithCutEBEE<OffEle,float>(iBooker,baseName+"_et",
							 baseName+" E_{T};E_{T} (GeV)",
							 bins.et.nr,bins.et.min,bins.et.max,&OffEle::et,cut));
  histVec.push_back(new MonElemWithCutEBEE<OffEle,float>(iBooker,baseName+"_eta",
							 baseName+" #eta;#eta",
							 bins.eta.nr,bins.eta.min,bins.eta.max,
							 &OffEle::detEta,cut ? cut->clone(): NULL));		
  histVec.push_back(new MonElemWithCutEBEE<OffEle,float>(iBooker,baseName+"_phi",
							 baseName+" #phi;#phi (rad)",
							 bins.phi.nr,bins.phi.min,bins.phi.max,
							 &OffEle::phi,cut ? cut->clone():NULL));		
  histVec.push_back(new MonElemWithCutEBEE<OffEle,int>(iBooker,baseName+"_nVertex",
							 baseName+" nVertex;nVertex",
							 bins.nVertex.nr,bins.nVertex.min,bins.nVertex.max,
							 &OffEle::NVertex,cut ? cut->clone():NULL));
  /*  histVec.push_back(new MonElemWithCutEBEE<OffEle,int>(iBooker,baseName+"_charge",
						       baseName+" Charge; charge",
						       bins.charge.nr,bins.charge.min,bins.charge.max,
						       &OffEle::charge,cut ? cut->clone():NULL)); */
}


void MonElemFuncs::initStdPhoCutHists(std::vector<MonElemWithCutBase<OffPho>*>& histVec,const std::string& filterName,const std::string& baseName,const BinData& bins,EgHLTDQMCut<OffPho>* cut)
{
  histVec.push_back(new MonElemWithCutEBEE<OffPho,float>(iBooker,baseName+"_et",
							 baseName+" E_{T};E_{T} (GeV)",
							 bins.et.nr,bins.et.min,bins.et.max,&OffPho::et,cut));
  histVec.push_back(new MonElemWithCutEBEE<OffPho,float>(iBooker,baseName+"_eta",
							 baseName+" #eta;#eta",
							 bins.eta.nr,bins.eta.min,bins.eta.max,
							 &OffPho::detEta,cut ? cut->clone(): NULL));		
  histVec.push_back(new MonElemWithCutEBEE<OffPho,float>(iBooker,baseName+"_phi",
							 baseName+" #phi;#phi (rad)",
							 bins.phi.nr,bins.phi.min,bins.phi.max,
							 &OffPho::phi,cut ? cut->clone():NULL));
  /* histVec.push_back(new MonElemWithCutEBEE<OffPho,int>(iBooker,baseName+"_charge",
						       baseName+" Charge; charge",
						       bins.charge.nr,bins.charge.min,bins.charge.max,
						       &OffPho::charge,cut ? cut->clone():NULL)); */
}

//we transfer ownership of eleCut to addTrigLooseTrigHist which transfers it to the eleMonElems
void MonElemFuncs::initTightLooseTrigHists(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffEle>* eleCut)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];
    //this step is necessary as we want to transfer ownership of eleCut to the addTrigLooseTrigHist func on the last iteration
    //but clone it before that
    //perhaps my object ownership rules need to be re-evalulated
    if(trigNr!=tightLooseTrigs.size()-2) addTightLooseTrigHist(eleMonElems,tightTrig,looseTrig,eleCut->clone(),"gsfEle",bins);
    else addTightLooseTrigHist(eleMonElems,tightTrig,looseTrig,eleCut,"gsfEle",bins);
  }
} 


void MonElemFuncs::initTightLooseTrigHists(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins,EgHLTDQMCut<OffPho>* phoCut)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];

    //this step is necessary as we want to transfer ownership of phoCut to the addTrigLooseTrigHist func on the last iteration
    //but clone it before that
    //perhaps my object ownership rules need to be re-evalulated
    if(trigNr!=tightLooseTrigs.size()-2) addTightLooseTrigHist(phoMonElems,tightTrig,looseTrig,phoCut->clone(),"pho",bins);
    else addTightLooseTrigHist(phoMonElems,tightTrig,looseTrig,phoCut,"pho",bins);
  }
}



   
//there is nothing electron specific here, will template at some point
void MonElemFuncs::addTightLooseTrigHist(std::vector<MonElemContainer<OffEle>*>& eleMonElems,
					 const std::string& tightTrig,const std::string& looseTrig,
					 EgHLTDQMCut<OffEle>* eleCut,
					 const std::string& histId,const BinData& bins)
{
  MonElemContainer<OffEle>* passMonElem = NULL;
  passMonElem = new MonElemContainer<OffEle>(tightTrig+"_"+looseTrig+"_"+histId+"_passTrig","",
					     &(*(new EgMultiCut<OffEle>) << 
					       new EgObjTrigCut<OffEle>(trigCodes.getCode(tightTrig+":"+looseTrig),EgObjTrigCut<OffEle>::AND)  <<
					       eleCut->clone()));
  
  
  MonElemContainer<OffEle>* failMonElem = NULL;
  failMonElem = new MonElemContainer<OffEle>(tightTrig+"_"+looseTrig+"_"+histId+"_failTrig","",
					     &(*(new EgMultiCut<OffEle>) << 
					       new EgObjTrigCut<OffEle>(trigCodes.getCode(looseTrig),EgObjTrigCut<OffEle>::AND,trigCodes.getCode(tightTrig))  << 
					       eleCut));
  
  MonElemFuncs::initStdEleHists(passMonElem->monElems(),tightTrig+"_"+looseTrig,passMonElem->name(),bins);
  MonElemFuncs::initStdEleHists(failMonElem->monElems(),tightTrig+"_"+looseTrig,failMonElem->name(),bins);
  eleMonElems.push_back(passMonElem);
  eleMonElems.push_back(failMonElem);
}

 
//there is nothing photon specific here, will template at some point
void MonElemFuncs::addTightLooseTrigHist(std::vector<MonElemContainer<OffPho>*>& phoMonElems,
					 const std::string& tightTrig,const std::string& looseTrig,
					 EgHLTDQMCut<OffPho>* phoCut,
					 const std::string& histId,const BinData& bins)
{
  MonElemContainer<OffPho>* passMonElem = NULL;
  passMonElem = new MonElemContainer<OffPho>(tightTrig+"_"+looseTrig+"_"+histId+"_passTrig","",
					     &(*(new EgMultiCut<OffPho>) << 
					       new EgObjTrigCut<OffPho>(trigCodes.getCode(tightTrig+":"+looseTrig),EgObjTrigCut<OffPho>::AND)  <<
					       phoCut->clone()));
  
  
  MonElemContainer<OffPho>* failMonElem = NULL;
  failMonElem = new MonElemContainer<OffPho>(tightTrig+"_"+looseTrig+"_"+histId+"_failTrig","",
					     &(*(new EgMultiCut<OffPho>) << 
					       new EgObjTrigCut<OffPho>(trigCodes.getCode(looseTrig),EgObjTrigCut<OffPho>::AND,trigCodes.getCode(tightTrig))  << 
					       phoCut));
  
  MonElemFuncs::initStdPhoHists(passMonElem->monElems(),tightTrig+"_"+looseTrig,passMonElem->name(),bins);
  MonElemFuncs::initStdPhoHists(failMonElem->monElems(),tightTrig+"_"+looseTrig,failMonElem->name(),bins);
  phoMonElems.push_back(passMonElem);
  phoMonElems.push_back(failMonElem);
}



//we transfer ownership of eleCut to the monitor elements
void MonElemFuncs::initTightLooseTrigHistsTrigCuts(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];
    EgHLTDQMCut<OffEle>* eleCut = new EgHLTDQMUserVarCut<OffEle,TrigCodes::TrigBitSet>(&OffEle::trigCutsCutCode,trigCodes.getCode(tightTrig));
    addTightLooseTrigHist(eleMonElems,tightTrig,looseTrig,eleCut,"gsfEle_trigCuts",bins);
  }
} 

//we transfer ownership of phoCut to the monitor elements
void MonElemFuncs::initTightLooseTrigHistsTrigCuts(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];
    EgHLTDQMCut<OffPho>* phoCut = new EgHLTDQMUserVarCut<OffPho,TrigCodes::TrigBitSet>(&OffPho::trigCutsCutCode,trigCodes.getCode(tightTrig));
    addTightLooseTrigHist(phoMonElems,tightTrig,looseTrig,phoCut,"pho_trigCuts",bins);
  }
} 


//we transfer ownership of eleCut to the monitor elements
void MonElemFuncs::initTightLooseDiObjTrigHistsTrigCuts(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];
    EgHLTDQMCut<OffEle>* eleCut = new EgDiEleUserCut<TrigCodes::TrigBitSet>(&OffEle::trigCutsCutCode,trigCodes.getCode(tightTrig));
    addTightLooseTrigHist(eleMonElems,tightTrig,looseTrig,eleCut,"gsfEle_trigCuts",bins);
  }
} 


//we transfer ownership of phoCut to the monitor elements
void MonElemFuncs::initTightLooseDiObjTrigHistsTrigCuts(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string>& tightLooseTrigs,const BinData& bins)
{
  for(size_t trigNr=0;trigNr<tightLooseTrigs.size();trigNr++){
    std::vector<std::string> splitString;
    boost::split(splitString,tightLooseTrigs[trigNr],boost::is_any_of(std::string(":")));
    if(splitString.size()!=2) continue; //format incorrect
    const std::string& tightTrig = splitString[0];
    const std::string& looseTrig = splitString[1];
    EgHLTDQMCut<OffPho>* phoCut = new EgDiPhoUserCut<TrigCodes::TrigBitSet>(&OffPho::trigCutsCutCode,trigCodes.getCode(tightTrig));
    addTightLooseTrigHist(phoMonElems,tightTrig,looseTrig,phoCut,"pho_trigCuts",bins);
  }
}



//tag and probe trigger efficiencies
//this is to measure the trigger efficiency with respect to a fully selected offline electron
//using a tag and probe technique (note: this will be different to the trigger efficiency normally calculated) 
void MonElemFuncs::initTrigTagProbeHists(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::vector<std::string> filterNames,int cutMask,const BinData& bins)
{
  for(size_t filterNr=0;filterNr<filterNames.size();filterNr++){ 
    
    std::string trigName(filterNames[filterNr]);
    //  float etCutValue = trigTools::getSecondEtThresFromName(trigName);
    float etCutValue = 0.;
    //std::cout<<"TrigName= "<<trigName<<"   etCutValue= "<<etCutValue<<std::endl;
    MonElemContainer<OffEle>* monElemCont = new MonElemContainer<OffEle>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut_New(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFiltesr"),cutMask,&OffEle::cutCode));
    //this is all that pass trigtagprobecut
    MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_all",bins,new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC));
    //this is all that pass trigtagprobecut and the probe passes the test trigger
    MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_pass",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is NOT a tag
    MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_passNotTag",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND,trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffEle>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is also a tag
    MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_passTagTag",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND) <<  new EgObjTrigCut<OffEle>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffEle>::AND) ));
    //this is all that pass trigtagprobecut and the probe fails the test trigger
    MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_fail",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),EgObjTrigCut<OffEle>::AND,trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));
    /*
    monElemCont->monElems().push_back(new MonElemMgrEBEE<OffEle,float>(iBooker,trigName+"_"+monElemCont->name()+"_gsfEle_all_etUnCut",monElemCont->name()+"_gsfEle_all E_{T} (Uncut);E_{T} (GeV)",
								       bins.et.nr,bins.et.min,bins.et.max,&OffEle::et));
    monElemCont->cutMonElems().push_back(new MonElemWithCutEBEE<OffEle,float>(iBooker,trigName+"_"+monElemCont->name()+"_gsfEle_pass_etUnCut",monElemCont->name()+"_gsfEle_pass E_{T} (Uncut);E_{T} (GeV)",
    bins.et.nr,bins.et.min,bins.et.max,&OffEle::et,new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));*/
    eleMonElems.push_back(monElemCont);
  } //end filter names loop
   
}

//Only one at a time so I can set the folder
void MonElemFuncs::initTrigTagProbeHist(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::string filterName,int cutMask,const BinData& bins)
{   
  std::string trigName(filterName);
  //float etCutValue = 1.1*trigTools::getSecondEtThresFromName(filterName);
  float etCutValue = 0.;
  //std::cout<<"TrigName= "<<trigName<<"   etCutValue= "<<etCutValue<<std::endl;
  MonElemContainer<OffEle>* monElemCont = new MonElemContainer<OffEle>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut_New(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),cutMask,&OffEle::cutCode));
  //this is all that pass trigtagprobecut
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_all",bins,new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC));
  //this is all that pass trigtagprobecut and the probe passes the test trigger
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_pass",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));
  //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is NOT a tag
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_passNotTag",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND,trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffEle>::AND)));
  //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is also a tag
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_passTagTag",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND) <<  new EgObjTrigCut<OffEle>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffEle>::AND) ));
  //this is all that pass trigtagprobecut and the probe fails the test trigger
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_gsfEle_fail",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),EgObjTrigCut<OffEle>::AND,trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));
  /*
    monElemCont->monElems().push_back(new MonElemMgrEBEE<OffEle,float>(iBooker,trigName+"_"+monElemCont->name()+"_gsfEle_all_etUnCut",monElemCont->name()+"_gsfEle_all E_{T} (Uncut);E_{T} (GeV)",
    bins.et.nr,bins.et.min,bins.et.max,&OffEle::et));
    monElemCont->cutMonElems().push_back(new MonElemWithCutEBEE<OffEle,float>(iBooker,trigName+"_"+monElemCont->name()+"_gsfEle_pass_etUnCut",monElemCont->name()+"_gsfEle_pass E_{T} (Uncut);E_{T} (GeV)",
    bins.et.nr,bins.et.min,bins.et.max,&OffEle::et,new EgObjTrigCut<OffEle>(trigCodes.getCode(trigName),EgObjTrigCut<OffEle>::AND)));*/
  eleMonElems.push_back(monElemCont);
}


void MonElemFuncs::initTrigTagProbeHist_2Leg(std::vector<MonElemContainer<OffEle>*>& eleMonElems,const std::string filterName,int cutMask,const BinData& bins)
{  
 
  std::string trigNameLeg1 = filterName.substr(0,filterName.find("::")); 
  std::string trigNameLeg2 = filterName.substr(filterName.find("::")+2);

  float etCutValue = 0.;
  MonElemContainer<OffEle>* monElemCont = new MonElemContainer<OffEle>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut_New(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),cutMask,&OffEle::cutCode));
  //this is all that pass trigtagprobecut
  //MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigNameLeg2,trigNameLeg2+"_"+monElemCont->name()+"_gsfEle_allEt20",bins,new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC));
  //this is all that pass trigtagprobecut and the probe passes the first trigger
  //MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigNameLeg2,trigNameLeg2+"_"+monElemCont->name()+"_gsfEle_passEt20",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigNameLeg1),EgObjTrigCut<OffEle>::AND)));
  //this is all that pass trigtagprobecut and the probe passes the second trigger and fails the first trigger
  MonElemFuncs::initStdEleCutHists(monElemCont->cutMonElems(),trigNameLeg2,trigNameLeg2+"_"+monElemCont->name()+"_gsfEle_passLeg2failLeg1",bins,&(*(new EgMultiCut<OffEle>) << new EgGreaterCut<OffEle,float>(etCutValue,&OffEle::etSC) << new EgObjTrigCut<OffEle>(trigCodes.getCode(trigNameLeg2),EgObjTrigCut<OffEle>::AND,trigCodes.getCode(trigNameLeg1),EgObjTrigCut<OffEle>::AND)));

}


//Now same for photons
void MonElemFuncs::initTrigTagProbeHists(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::vector<std::string> filterNames,int cutMask,const BinData& bins)
{
  for(size_t filterNr=0;filterNr<filterNames.size();filterNr++){ 
    
    std::string trigName(filterNames[filterNr]);
    //float etCutValue = trigTools::getSecondEtThresFromName(trigName);
    float etCutValue = 0.;
    //std::cout<<"TrigName= "<<trigName<<"   etCutValue= "<<etCutValue<<std::endl;
    MonElemContainer<OffPho>* monElemCont = new MonElemContainer<OffPho>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut_NewPho(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),cutMask,&OffPho::cutCode));
    //this is all that pass trigtagprobecut
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_all",bins,new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC));
    //this is all that pass trigtagprobecut and the probe passes the test trigger
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_pass",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is NOT a tag
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_passNotTag",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND,trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffPho>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is also a tag
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_passTagTag",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND) <<  new EgObjTrigCut<OffPho>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffPho>::AND) ));
    //this is all that pass trigtagprobecut and the probe fails the test trigger
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_fail",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),EgObjTrigCut<OffPho>::AND,trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));
    /*
    monElemCont->monElems().push_back(new MonElemMgrEBEE<OffPho,float>(iBooker,trigName+"_"+monElemCont->name()+"_pho_all_etUnCut",monElemCont->name()+"_gsfEle_all E_{T} (Uncut);E_{T} (GeV)",
								       bins.et.nr,bins.et.min,bins.et.max,&OffPho::et));
    monElemCont->cutMonElems().push_back(new MonElemWithCutEBEE<OffPho,float>(iBooker,trigName+"_"+monElemCont->name()+"_pho_pass_etUnCut",monElemCont->name()+"_gsfEle_pass E_{T} (Uncut);E_{T} (GeV)",
    bins.et.nr,bins.et.min,bins.et.max,&OffPho::et,new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));*/
    phoMonElems.push_back(monElemCont);
  } //end filter names loop
   
}

void MonElemFuncs::initTrigTagProbeHist(std::vector<MonElemContainer<OffPho>*>& phoMonElems,const std::string filterName,int cutMask,const BinData& bins)
{
    std::string trigName(filterName);
    //float etCutValue = 1.1*trigTools::getSecondEtThresFromName(trigName);
    float etCutValue = 0.;
    //std::cout<<"TrigName= "<<trigName<<"   etCutValue= "<<etCutValue<<std::endl;
    MonElemContainer<OffPho>* monElemCont = new MonElemContainer<OffPho>("trigTagProbe","Trigger Tag and Probe",new EgTrigTagProbeCut_NewPho(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),cutMask,&OffPho::cutCode));
    //this is all that pass trigtagprobecut
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_all",bins,new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC));
    //this is all that pass trigtagprobecut and the probe passes the test trigger
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_pass",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is NOT a tag
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_passNotTag",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND,trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffPho>::AND)));
    //this is all that pass trigtagprobecut and the probe passes the test trigger and the probe is also a tag
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_passTagTag",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND) <<  new EgObjTrigCut<OffPho>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17TrackIsolFilter"),EgObjTrigCut<OffPho>::AND) ));
    //this is all that pass trigtagprobecut and the probe fails the test trigger
    MonElemFuncs::initStdPhoCutHists(monElemCont->cutMonElems(),trigName,trigName+"_"+monElemCont->name()+"_pho_fail",bins,&(*(new EgMultiCut<OffPho>) << new EgGreaterCut<OffPho,float>(etCutValue,&OffPho::etSC) << new EgObjTrigCut<OffPho>(trigCodes.getCode("hltEle32CaloIdTCaloIsoTTrkIdTTrkIsoTSC17HEDoubleFilter"),EgObjTrigCut<OffPho>::AND,trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));
    /*
    monElemCont->monElems().push_back(new MonElemMgrEBEE<OffPho,float>(iBooker,trigName+"_"+monElemCont->name()+"_pho_all_etUnCut",monElemCont->name()+"_gsfEle_all E_{T} (Uncut);E_{T} (GeV)",
								       bins.et.nr,bins.et.min,bins.et.max,&OffPho::et));
    monElemCont->cutMonElems().push_back(new MonElemWithCutEBEE<OffPho,float>(iBooker,trigName+"_"+monElemCont->name()+"_pho_pass_etUnCut",monElemCont->name()+"_gsfEle_pass E_{T} (Uncut);E_{T} (GeV)",
    bins.et.nr,bins.et.min,bins.et.max,&OffPho::et,new EgObjTrigCut<OffPho>(trigCodes.getCode(trigName),EgObjTrigCut<OffPho>::AND)));*/
    phoMonElems.push_back(monElemCont);
}




