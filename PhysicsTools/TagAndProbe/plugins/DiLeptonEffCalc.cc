
// /*   _Anil Pratap Singh, */
// /*   _Panjab University */

// /* Calculates the mean sample efficiency */
// /* for a dilepton selection. */


// /* ********************************************************* */
// /*  This analyzer module can run over an edm::TTree produced by 
// /*  TagProbeEDMNtuple class and then calculate the mean 
// /*  reconstruction efficiency (including offline and/or online) 
// /*  and statistical uncertainty therein. If you want to run this 
// /*  analyzer over a cms:RECO dataset collection, you may need to
// /*  make few minor changes to it.
// /* ********************************************************* */


/* // system include files */
#include <memory>
#include<fstream>
#include<map>
#include<cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TagAndProbe/interface/EffTableLoader.h"
#include "PhysicsTools/TagAndProbe/interface/dibin.h"
//
// class decleration
//

EffTableLoader* tab1_Ofline;
EffTableLoader* tab2_Ofline;
EffTableLoader* tab1_Online;
EffTableLoader* tab2_Online;

typedef std::map<dibin, int> dibinIntMap;
typedef std::map<int, float> binMap;
typedef std::map<dibin, float> dibinFloatMap;
class DiLepEffCalc : public edm::EDAnalyzer {
 public:
  explicit DiLepEffCalc(const edm::ParameterSet&);
  ~DiLepEffCalc();
  
  
 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  dibin SetAppropriateKey(int i,int j, int k,int l, int flag);

  void FillMapEntry(int* indices, float* Effi,
		    float* Err,  binMap& EffId_i,
		    binMap& EffId_j,binMap& EffTr_i,
		    binMap& EffTr_k,dibinIntMap& CntMap,
		    dibinFloatMap& EffMap);

  int TotalOccup(dibinIntMap& OccMap);
  binMap calculateErrorCoff(binMap&, binMap&, binMap&, binMap&, 
			    dibinIntMap&,int);
  float calculateErrorTerm(binMap Coff, binMap Err);
  float  MeanEff(dibinFloatMap EffMap, dibinIntMap OccMap); 
 
  dibinIntMap CollectTheseElements(dibinIntMap& Occ, int ind, int flag);
  std::vector<float> FindEntry(EffTableLoader& Eff, 
			       float pt, float eta);

  // ----------member data ---------------------------
  std::ofstream ff;
  std::vector<std::string> tablenames_;
  std::string outTextName_;
  int tagProbeType_;
  double ptMinCut_, ptMaxCut_, etaMaxCut_;
  float effId_first,effId_seknd,errId_first,errId_seknd;
  float effTr_first,effTr_seknd,errTr_first,errTr_seknd;
};


typedef  std::map<dibin, float> dibinFloatMap;
typedef  std::map<dibin, int> dibinIntMap;
typedef  std::map<int, float> binMap;





 dibinIntMap CntMap;//Carry diBin Occupiancies
 dibinFloatMap EffMap;// Carry diBin Efficiency values
 binMap EffId_i;//Carry EffiId values for first lep
 binMap EffId_j;//Carry EffiId values for Seknd lep
 binMap ErrId_i;//Carry ErrId  values for first lep
 binMap ErrId_j;//Carry ErrId  values foe Seknd lep
 binMap EffTr_k;//Carry EffiTr values for first lep
 binMap EffTr_l;//Carry EffiTr values for Seknd lep
 binMap ErrTr_k;//Carry ErrTr  values for first lep
 binMap ErrTr_l;//Carry ErrTr  values foe Seknd lep






//===================================================================
//  ***** Important: The users must supply the table names in the  
//        following order:
//        ele-1-offline, ele-1-online, ele-2-offline, ele-2-online
//   In case table is absent, we set: efficiency = 1, error = 0.0
//======================================================================


DiLepEffCalc::DiLepEffCalc(const edm::ParameterSet& iConfig){

  tablenames_   = iConfig.getParameter< std::vector<std::string> >("TableNames"); 
  tagProbeType_ = iConfig.getUntrackedParameter< int >("TagProbeType",0);
  outTextName_  = iConfig.getUntrackedParameter< std::string >("OutputTextFileName", "Efficiency.txt");
  ptMinCut_     = iConfig.getUntrackedParameter< double >("PtMinCut", 20.0);
  ptMaxCut_     = iConfig.getUntrackedParameter< double >("PtMaxCut", 100.0);
  etaMaxCut_    = iConfig.getUntrackedParameter< double >("EtaMaxCut", 2.5);

  if(tablenames_.size()==0) {
     edm::LogError("DiLeptonEffCalc") <<
      "Fatal Error: You must supply at least one efficiency table !";
    exit(1);
  }

  if(tagProbeType_<0 || tagProbeType_>4){
    edm::LogError("DiLeptonEffCalc") <<
      "Fatal Error: Tag-Probe Type can only be in the range 0--4 !";
    exit(1);
  }
  
  if(tablenames_.size()>=1) tab1_Ofline = new EffTableLoader(tablenames_[0]);
  if(tablenames_.size()>=2) tab1_Online = new EffTableLoader(tablenames_[1]);
  if(tablenames_.size()>=3) tab2_Ofline = new EffTableLoader(tablenames_[2]);
  if(tablenames_.size()==4) tab2_Online = new EffTableLoader(tablenames_[3]);
 
}
//===================================================================




//===================================
DiLepEffCalc::~DiLepEffCalc(){

  if( tab1_Ofline ) delete tab1_Ofline;
  if( tab2_Ofline ) delete tab2_Ofline;
  if( tab1_Online ) delete tab1_Online;
  if( tab2_Online ) delete tab2_Online;
}
//===================================




//=====================================================
void DiLepEffCalc::beginJob(){}
//=====================================================





//================================================================

void  DiLepEffCalc::analyze(const edm::Event& iEvent, 
			    const edm::EventSetup& iSetup)
{

  //define handles
  //get values of Pt and eta.

  edm::Handle< std::vector<float> > probe_pt;
  iEvent.getByLabel("TPEdm","TPProbept", probe_pt);

  edm::Handle< std::vector<float> > probe_eta;
  iEvent.getByLabel("TPEdm","TPProbeeta", probe_eta);

  edm::Handle< std::vector<float> > tag_pt;
  iEvent.getByLabel("TPEdm","TPTagpt", tag_pt);

  edm::Handle< std::vector<float> > tag_eta;
  iEvent.getByLabel("TPEdm","TPTageta", tag_eta);

  if( (*tag_pt).size()==0 ||  (*tag_eta).size()==0 || 
      (*probe_pt).size()==0 || (*probe_eta).size()==0) return;



  //************** get variable values for the right TagProbeType 
  float pt1 =  (*tag_pt)[tagProbeType_];
  float eta1 =  (*tag_eta)[tagProbeType_];
  float pt2 = (*probe_pt)[tagProbeType_];
  float eta2 = (*probe_eta)[tagProbeType_];
 


  //************** check for acceptance cut
  if( pt1<ptMinCut_ || pt2<ptMinCut_ || 
      pt1>ptMaxCut_ || pt2>ptMaxCut_ || 
      fabs(eta1)>etaMaxCut_ || fabs(eta2)>etaMaxCut_ ||
      (fabs(eta1)>1.4442 && fabs(eta1)<1.560) || 
      (fabs(eta2)>1.4442 && fabs(eta2)<1.560) ) return;



  //************* electron 1 : offline efficiency
  int firstId = 0;
  if(tab1_Ofline) {
    std::vector<float> EffInfo1_ofline =  FindEntry( *tab1_Ofline, pt1, eta1);
    effId_first = EffInfo1_ofline[1];
    errId_first = EffInfo1_ofline[2];
    firstId  = (int) EffInfo1_ofline[3] ;
  }
  else {
    effId_first = 1.0;
    errId_first = 0.0;
  }  
  edm::LogInfo("DiLeptonEffCalc") << "electron-1 offline eff = " << effId_first << "  +/-  " << errId_first;
      


  //************* electron 1 : online efficiency
  int firstTr = 0;
  if(tab1_Online) {
    std::vector<float> EffInfo1_online =  FindEntry( *tab1_Online, pt1, eta1);
    effTr_first =  EffInfo1_online[1];
    errTr_first =  EffInfo1_online[2];
    firstTr     =(int) EffInfo1_online[3] ;
  }
  else {
    effTr_first =  1.0;
    errTr_first =  0.0;
  }
  edm::LogInfo("DiLeptonEffCalc") << "electron-1 online eff = " << effTr_first << "  +/-  " << errTr_first;



  //************* electron 2 : offline efficiency
  int sekndId = 0;
  if(tab2_Ofline) {
    std::vector<float> EffInfo2_ofline =  FindEntry( *tab2_Ofline, pt2, eta2);
    effId_seknd = EffInfo2_ofline[1];
    errId_seknd = EffInfo2_ofline[2];
    sekndId = (int) EffInfo2_ofline[3] ;
  }
  else {
    effId_seknd = 1.0;
    errId_seknd = 0.0;
  }
  edm::LogInfo("DiLeptonEffCalc") << "electron-2 offline eff = " << effId_seknd << "  +/-  " << errId_seknd;



  //************* electron 2 : online efficiency
  int sekndTr = 0;
  if(tab2_Online) {
    std::vector<float> EffInfo2_online =  FindEntry( *tab2_Online, pt2, eta2);
    effTr_seknd =EffInfo2_online[1];
    errTr_seknd =EffInfo2_online[2];
    sekndTr =(int) EffInfo2_online[3] ;
  }
  else {
    effTr_seknd = 1.0;
    errTr_seknd = 0.0;
  }
   edm::LogInfo("DiLeptonEffCalc") << "electron-2 online eff = " << effTr_seknd << "  +/-  " << errTr_seknd;    
  
  
    
  EffId_i[firstId]    = effId_first;
  ErrId_i[firstId]    = errId_first;
  EffTr_k[firstTr]    = effTr_first;
  ErrTr_k[firstTr]    = errTr_first;
      
  EffId_j[sekndId]    = effId_seknd; 
  ErrId_j[sekndId]    = errId_seknd;  
  EffTr_l[sekndTr]    = effTr_seknd;
  ErrTr_l[sekndTr]    = errTr_seknd;
      
  dibin index(firstId, sekndId, firstTr, sekndTr);
    
  if(CntMap.count(index) >0) CntMap[index]++;
  else {
    CntMap[index] = 1;
    EffMap[index]=effId_seknd*effId_first*( 1-(1-effTr_first)*(1-effTr_seknd));
  }           
}
//======================================================================











//======================================================================

void  DiLepEffCalc::endJob()
{
  float Efficiency =MeanEff(EffMap, CntMap);
  binMap Ti = calculateErrorCoff(EffId_i, EffId_j,EffTr_k,  EffTr_l,CntMap,0 );
  binMap Tj = calculateErrorCoff(EffId_j, EffId_i,EffTr_k,  EffTr_l,CntMap,1 );
  binMap Tk = calculateErrorCoff(EffTr_k, EffId_i,EffId_j,  EffTr_l,CntMap,2 );
  binMap Tl = calculateErrorCoff(EffTr_l, EffId_i,EffId_j,  EffTr_k,CntMap,3 );
  float i_term = calculateErrorTerm(Ti,ErrId_i );
  float j_term = calculateErrorTerm(Tj,ErrId_j );
  float k_term = calculateErrorTerm(Tk,ErrTr_k );
  float l_term = calculateErrorTerm(Tl,ErrTr_l );
  float Error  = std::sqrt(i_term+j_term+k_term+l_term);

  ff.open(outTextName_.c_str());
  ff<<"Number of 2-dimensional Efficiency bins: "<<EffMap.size()<<std::endl;
  ff<<"Number of Z candidates found: "<<TotalOccup(CntMap)<<std::endl;
  for(dibinIntMap::iterator it = CntMap.begin(); it !=CntMap.end(); it++)
    {
      int fu[4];
      (it->first).print(fu);
      float ptmid1=0.0, etamid1=-3.0, ptmid2=0.0, etamid2=-3.0;
      if(tab1_Ofline) { 
	ptmid1  = (tab1_Ofline->GetCellCenter(fu[0])).first;
	ptmid2  = (tab1_Ofline->GetCellCenter(fu[1])).first; 
	etamid1 = (tab1_Ofline->GetCellCenter(fu[0])).second; 
	etamid2 = (tab1_Ofline->GetCellCenter(fu[1])).second;
      }
      else {
      ff<< "Something is wrong with the input efficiency table information" << std::endl;
      }
      //**** print out the efficiency & occupancy in each bin *****
      ff<<""<<std::endl;
      ff << "(pT,eta) Bin Center for electrons 1 and 2:  [ ("  <<  ptmid1
	 << " , " << etamid1 << "), (" << ptmid2 << " , " << etamid2
	 << ") ]" << std::endl;
      ff<< "The Occupancy in this bin: "<<(it->second)<<std::endl;	
    }

  ff << "--------------------------------------------------------" << std::endl;
  ff << "The mean Sample Efficiency: " << Efficiency << 
    "    +/-   " << Error << std::endl;
  ff << "--------------------------------------------------------" << std::endl;
  ff.close();
}
//===========================================================







//===========================================================

binMap DiLepEffCalc::calculateErrorCoff(binMap& EffMap1,
					binMap& EffMap2,
					binMap& EffMap3,
					binMap& EffMap4, 
					dibinIntMap& OccMap,
					int flag){ 

  edm::LogInfo("DiLeptonEffCalc") << "The Coefficient Calc Begins =======";
  binMap Ti;
  dibinIntMap Tri, Trij, Trijk, Trijkl;  
  binMap::iterator BnIter1, BnIter2, BnIter3, BnIter4;
  int N = TotalOccup(OccMap);
  for(BnIter1 = EffMap1.begin(); BnIter1 != EffMap1.end(); BnIter1++) {
    int i = BnIter1->first;
    Tri = CollectTheseElements(OccMap, i, 1);
    for(BnIter2 = EffMap2.begin(); BnIter2 != EffMap2.end(); BnIter2++) {
      int j = BnIter2->first;
      Trij = CollectTheseElements(Tri,j, 2);
      for(BnIter3 = EffMap3.begin(); BnIter3 != EffMap3.end(); BnIter3++) {
	int k = BnIter3->first;
	Trijk = CollectTheseElements(Trij,k,3);
	for(BnIter4 = EffMap4.begin(); BnIter4 != EffMap4.end(); BnIter4++) {
	  int l = BnIter4->first;
	  dibin key=SetAppropriateKey(i, j, k, l, flag);
	  dibinIntMap::iterator cntIter=Trijk.find(key);
	  if((cntIter->first)==(key)){
	    if(flag==0||flag==1) 
	      Ti[i]= Ti[i]+(BnIter2->second)*
		(1-(1-BnIter3->second)*(1-BnIter4->second))*
		(cntIter->second)/N;
	    if(flag==2||flag==3) 
	      Ti[i]= Ti[i]+(BnIter2->second)*
		(BnIter3->second)*(BnIter4->second)*(cntIter->second)/N;   
	  }
	}
      }
    }
  }
  return Ti;
} 
//======================================================================






//==================================================================

dibinIntMap DiLepEffCalc:: CollectTheseElements(dibinIntMap& Occ, 
						int i, int flag)
{
  dibinIntMap temp;
  dibinIntMap::iterator it ;
  for(it = Occ.begin(); it != Occ.end(); it++)
    {
      dibin Key = (it->first);
      bool b1 =0;
      if(flag==1)b1= (Key.GetOuterIDKey()==i);
      if(flag==2)b1= (Key.GetInnerIDKey()==i);
      if(flag==3)b1= (Key.GetOuterTrKey()==i);
      if(flag==4)b1= (Key.GetInnerTrKey()==i);
      if(b1)	{
	int occup = it->second;
	temp[Key] = occup;
      }
    }
  return temp;
}
//==================================================================





//========================================================================

dibin DiLepEffCalc::SetAppropriateKey(int i, int j, int k, int l, int flag)
{
  dibin key(0,0,0,0);
       if(flag==0){dibin thisKey(i,j,k,l); key = thisKey;}
  else if(flag==1){dibin thisKey(j,i,k,l); key = thisKey;}
  else if(flag==2){dibin thisKey(k,i,j,l); key = thisKey;}
  else if(flag==3){dibin thisKey(l,i,j,k); key = thisKey;}
  return key;
}
//========================================================================






//========================================================================

float DiLepEffCalc::calculateErrorTerm(binMap Coff, binMap Err)
{  
  float term = 0;
  binMap::iterator BnIter1, BnIter2;
  for(BnIter1 = Coff.begin(); BnIter1 != Coff.end(); BnIter1++)
    {
      BnIter2 = Err.find(BnIter1->first);
      if((BnIter2->first)==(BnIter1->first))
	term = term+std::pow(((BnIter1->second)*(BnIter2->second)),2);
    }
  return term;
}
//========================================================================







//========================================================================

int DiLepEffCalc::TotalOccup(dibinIntMap& OccMap)
{
  int N =0;
  dibinIntMap::iterator cntIter;
  for(cntIter = OccMap.begin(); cntIter != OccMap.end(); cntIter++)
    {
      N = N+cntIter->second;
      dibinIntMap::iterator itt = OccMap.find(cntIter->first);
    }
      
  return N; 
   
}
//=========================================================================







//========================================================================

float DiLepEffCalc::MeanEff(dibinFloatMap EffMap, dibinIntMap OccMap){
     float Eff = 0;
     dibinFloatMap::iterator effIter;
     for(effIter = EffMap.begin(); effIter != EffMap.end(); effIter++){
	 dibinIntMap::iterator cntIter =  OccMap.find(effIter->first);
	 Eff = Eff+(effIter->second)*(cntIter->second);
     }
     float N = TotalOccup(OccMap);
     return (Eff/N); 
   }
//=========================================================================









//========================================================================

std::vector<float>  DiLepEffCalc::FindEntry(EffTableLoader& Eff, 
					    float pt, float eta)
{
  float index = (float) Eff.GetBandIndex(pt, eta);
  std::vector<float> Effi= Eff.correctionEff(pt, eta);
  Effi.push_back(index);
  return Effi; 
}  
//=========================================================================



DEFINE_FWK_MODULE(DiLepEffCalc);
