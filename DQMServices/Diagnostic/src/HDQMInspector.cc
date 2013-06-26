//---------------------------------------------------------//
//
//-- extract summary informations from historic DB --
//-- plot summary informations vs run number or vs detID --
//
//---------------------------------------------------------//
//---------------------------------------------------------//
// 
//  12-08-2008 - domenico.giordano@cern.ch 
//  12-06-2008 - anne-catherine.le.bihan@cern.ch 
//
//---------------------------------------------------------//

#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include <time.h>
#include <cassert>
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TString.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TAxis.h"
#include "TMath.h"
#include "TLegend.h"
#include "TVirtualPad.h"
#include "TFormula.h"
#include "TObjArray.h"
#include "TObjString.h"


void HDQMInspector::style()
{
  TStyle* theStyle= new TStyle();
  theStyle->SetOptStat(0);
  //gROOT->SetStyle("Plain");
  theStyle->SetOptStat(0);
  theStyle->SetOptFit(111);
  theStyle->SetStatFont(12);
  theStyle->SetStatBorderSize(1);
  theStyle->SetCanvasColor(0);
  theStyle->SetCanvasBorderMode(0);
  theStyle->SetPadBorderMode(0);
  theStyle->SetPadColor(0);
  theStyle->SetLineWidth(1);
  theStyle->SetLineStyle(2);
  theStyle->SetPalette(1);
  theStyle->SetMarkerStyle(20);
  theStyle->SetMarkerColor(2);
  theStyle->SetLabelSize(0.05,"y");
  theStyle->SetLabelSize(0.04,"x");
  theStyle->SetTitleFontSize(0.2);
  theStyle->SetTitleW(0.9);
  theStyle->SetTitleH(0.06);
  theStyle->SetPadLeftMargin(0.12);   
  theStyle->SetPadTopMargin(0.13);
  theStyle->cd();
}

void HDQMInspector::setDB(const std::string & DBName, const std::string & DBTag, const std::string & DBauth)
{
  if( DBName_==DBName && DBTag_==DBTag && DBauth_ == DBauth)
    return;

  DBName_= DBName;
  DBTag_= DBTag;
  DBauth_ = DBauth;

  std::cout << "Name of DB = "<< DBName << std::endl;
  std::cout << "DBTag = "<< DBTag << std::endl;
  std::cout << "DBauth = "<< DBauth << std::endl;
  std::cout <<std::endl;

  accessDB();
  
  fOutFile = new TFile( "historicDQM.root","RECREATE" );
  if (!fOutFile->IsOpen()) {
    std::cerr << "ERROR: cannot open output file" << std::endl;
    exit(1);
  }
  fOutFile->cd();
}

void HDQMInspector::accessDB()
{
  //double start, end;
  // start = clock();   
 
  if(Iterator!=0)
    delete Iterator;
  
  Iterator = new CondCachedIter<HDQMSummary>(); 

  std::cout << "creating connection" << std::endl;
  Iterator->create(DBName_,DBTag_,DBauth_);
  std::cout << "connection created" << std::endl;

  
  InitializeIOVList();
  //  end = clock();
  //  if(iDebug)
  //std::cout <<"Time Creation link with Database = " <<  ((double) (end - start)) << " (a.u.)" <<std::endl; 
}


void  HDQMInspector::setBlackList(std::string const & ListItems)
{

  // Run over entire input string
  for (std::string::const_iterator Pos = ListItems.begin(); Pos != ListItems.end(); ) {

    // The rest of the string
    std::string Remainder(Pos, ListItems.end());

    // This entry will be from the beginning of the remainder to either a ","
    // or the end of the string
    std::string Entry = Remainder.substr(0, Remainder.find(","));

    // If we find a "-" we know it's a blacklist range
    if ( Entry.find("-") ) {

      // Get the first and last runs from this range
      int const FirstRun = atoi( Entry.substr(0, Entry.find("-")).c_str() );
      int const LastRun  = atoi( Entry.substr(Entry.find("-")+1).c_str() );

      // If you entered it stupidly we're going to stop here.
      if (FirstRun > LastRun) {
        std::cerr << "ERROR: FirstRun > LastRun in blackList" << std::endl;
        exit(1);
      }

      // For now the simplest thing to do is fill in gaps including each end
      for (int i = FirstRun; i <= LastRun; ++i) {
        blackList.push_back(i);
      }

    } else {
      // If we didn't see a "-" just add it to the list
      blackList.push_back( atoi(Entry.c_str()) );
    }

    // This is to make sure we are in the correct position as we go on.
    Pos += Entry.size();
    if (Pos != ListItems.end()) {
      Pos += 1;
    }

  }

  // sort the list for faster searching later
  std::sort(blackList.begin(), blackList.end());

  return;
}

void  HDQMInspector::setWhiteList(std::string const & ListItems)
{

  // Run over entire input string
  for (std::string::const_iterator Pos = ListItems.begin(); Pos != ListItems.end(); ) {

    // The rest of the string
    std::string Remainder(Pos, ListItems.end());

    // This entry will be from the beginning of the remainder to either a ","
    // or the end of the string
    std::string Entry = Remainder.substr(0, Remainder.find(","));

    // If we find a "-" we know it's a whitelist range
    if ( Entry.find("-") ) {

      // Get the first and last runs from this range
      int const FirstRun = atoi( Entry.substr(0, Entry.find("-")).c_str() );
      int const LastRun  = atoi( Entry.substr(Entry.find("-")+1).c_str() );

      // If you entered it stupidly we're going to stop here.
      if (FirstRun > LastRun) {
        std::cerr << "ERROR: FirstRun > LastRun in WhiteList" << std::endl;
        exit(1);
      }

      // For now the simplest thing to do is fill in gaps including each end
      for (int i = FirstRun; i <= LastRun; ++i) {
        whiteList.push_back(i);
      }

    } else {
      // If we didn't see a "-" just add it to the list
      whiteList.push_back( atoi(Entry.c_str()) );
    }

    // This is to make sure we are in the correct position as we go on.
    Pos += Entry.size();
    if (Pos != ListItems.end()) {
      Pos += 1;
    }

  }

  // sort the list for faster searching later
  std::sort(whiteList.begin(), whiteList.end());

  return;
}

std::string HDQMInspector::readListFromFile(const std::string & listFileName)
{
  std::ifstream listFile;
  listFile.open(listFileName.c_str());
  std::string listString;
  if( !listFile ) {
    std::cout << "Warning: list file" << listFileName << " not found" << std::endl;
    return listString;
  }
  while( !listFile.eof() ) {
    std::string line;
    listFile >> line;
    if( line != "" ) {
      listString += line;
      listString += ",";
    }
  }
  // Remove the last ","
  std::string::size_type pos = listString.find_last_of(",");
  if( pos != std::string::npos ) {
    listString.erase(pos);
  }
  std::cout << "whiteList = " << listString << std::endl;
  return listString;
}

bool  HDQMInspector::isListed(unsigned int run, std::vector<unsigned int>& vList)
{
  // This routine expectes a sorted list and returns true if the run is in the list,
  // false otherwise

  // Binary search is much faster, but you MUST give it a sorted list.
  if (std::binary_search(vList.begin(), vList.end(), run)) {
    if(iDebug) {
      std::cout << "\n Run "<< run << " is listed !!\n" << std::endl;
    }
    return true;
  }

  return false;

}


void HDQMInspector::InitializeIOVList()
{
  const HDQMSummary* reference;
  while((reference = Iterator->next())) {
    iovList.push_back(Iterator->getStartTime());
    if (iDebug) {
      std::cout << "iovList " << iovList.back() << std::endl;
    }
  } 
  Iterator->rewind();
}

bool HDQMInspector::setRange(unsigned int& firstRun, unsigned int& lastRun)
{
  unsigned int first,last;

  for(size_t i=0;i<iovList.size();++i) {
    if (iDebug) {
      std::cout << iovList.at(i)<< std::endl;
    }
  }

  std::vector<unsigned int>::iterator iter;

  iter=std::lower_bound(iovList.begin(),iovList.end(),firstRun);
  if (iter!=iovList.end())
    first=*iter;
  else{
    std::cout << "firstRun (" << firstRun << ") > last iov ("<<iovList.back()<< ")"<<std::endl; 
    return false;
  }

  iter=std::lower_bound(iovList.begin(),iovList.end(),lastRun);
  if (iter!=iovList.end()){
    if (*iter>lastRun) last = *(iter-1);
    else last=*iter;
  }
  else{
    last=iovList.back();
  }
  
  firstRun=first;
  lastRun=last; 
  std::cout << "setting Range firstRun (" << first << ") - lastRun ("<<last<< ")"<<std::endl; 
  Iterator->setRange(first,last);
  
  return true;
}

void HDQMInspector::createTrendLastRuns(const std::string ListItems, const std::string CanvasName,
                                        const int logy, const std::string Conditions, std::string const& Labels, const unsigned int nRuns, int const UseYRange, double const& YMin, double const& YMax)
{
  unsigned int first,last;
  unsigned int iovListSize = iovList.size();

  if (iovListSize>0) 
  { 
    last = iovList.back();

    if (iovListSize>=nRuns) {
      first = iovList.at(iovListSize-nRuns);
    } else {
      first = *iovList.begin();
    }
  }
  else return;

  createTrend(ListItems,CanvasName,logy,Conditions,Labels,first,last, UseYRange, YMin, YMax);

  return;
}

void HDQMInspector::createTrend(std::string ListItems, std::string CanvasName, int logy, std::string Conditions, std::string const& Labels, unsigned int firstRun, unsigned int lastRun, int const UseYRange, double const& YMin, double const& YMax)
{
  std::cout << "\n****************\nCreateTrend\n****************\n" << std::endl;
  std::cout << "ListItems : " << ListItems << std::endl;
  std::cout << "Conditions : " << Conditions << std::endl;

  vRun_.clear();
  vSummary_.clear();
  vDetIdItemList_.clear();

  std::vector<DetIdItemList> vDetIdItemListCut;
  
  size_t nPads=unpackItems(ListItems);   

  unpackConditions(Conditions,vDetIdItemListCut);
 
  //   double start = clock(); 

  std::cout << "firstRun " << firstRun << " lastRun " << lastRun << std::endl;
  if(!setRange(firstRun,lastRun)){
    Iterator->rewind();
    return;
  }
  const HDQMSummary* reference;
  while((reference = Iterator->next())) { 


    // Check the run and black and white lists
    if(Iterator->getStartTime()<firstRun || Iterator->getStartTime()>lastRun || isListed(reference->getRunNr(), blackList)) {
      continue;
    }
    if (whiteList.size() > 0 && !isListed(reference->getRunNr(), whiteList)) {
      continue;
    }

    if(vDetIdItemListCut.size()){
      for(size_t ij=0;ij!=vDetIdItemListCut.size();++ij){
        vDetIdItemListCut[ij].values=reference->getSummaryObj(vDetIdItemListCut[ij].detid, vDetIdItemListCut[ij].items);
      }

      if(!ApplyConditions(Conditions,vDetIdItemListCut))
        continue;
    }

    vRun_.push_back(reference->getRunNr());

    for(size_t ij=0;ij!=vDetIdItemList_.size();++ij){
      vDetIdItemList_[ij].values=reference->getSummaryObj(vDetIdItemList_[ij].detid, vDetIdItemList_[ij].items);
     
      vSummary_.insert(vSummary_.end(),vDetIdItemList_[ij].values.begin(),vDetIdItemList_[ij].values.end());   
      if(iDebug){
        std::cout << ListItems  << " run " << vRun_.back() << " values \n" ;
        DetIdItemList detiditemlist=vDetIdItemList_[ij];
        for(size_t i=0;i<detiditemlist.items.size();++i) {
          std::cout << "\t" << detiditemlist.items[i] << " " << detiditemlist.values[i] <<" " << i << " \n";
        }
        std::cout << "\n" << std::endl;
      }
    }
  }

  if(vRun_.size()) {
    plot(nPads, CanvasName, logy, Labels, UseYRange, YMin, YMax);
  }
   
     
  std::cout << "\n****** Ignore this error *****\n" << std::endl;
  Iterator->rewind();
  std::cout << "\n******************************\n" << std::endl;
}

void HDQMInspector::plot(size_t& nPads, std::string CanvasName, int logy, std::string const& Labels, int const UseYRange, double const YMin, double const YMax)
{
  std::cout << "\n********\nplot\n*****\n"<< std::endl;

  style();

  double *X, *Y, *EX, *EY, *YCumul;
  X=new double[vRun_.size()];
  Y=new double[vRun_.size()];
  EX=new double[vRun_.size()]; 
  EY=new double[vRun_.size()];  
  YCumul=new double[vRun_.size()];
  
  size_t index;
  TCanvas *C;
  TGraphErrors *graph;

  if(CanvasName==""){
    char name[128];
    sprintf(name,"%d",(int) clock());
    CanvasName=std::string(name);
  }
  
  std::string rootCName = CanvasName;
  rootCName.replace(rootCName.find("."),rootCName.size()-rootCName.find("."),"");
 
  C=new TCanvas(rootCName.c_str(),"");
  int ndiv=(int) sqrt(nPads);
  C->Divide(ndiv,nPads/ndiv+ (nPads%ndiv?1:0));
 
  int padCount=0;

  vlistItems_.clear();
  vdetId_.clear();

  for(size_t ic=0;ic<vDetIdItemList_.size();++ic){
    vlistItems_.insert(vlistItems_.end(),vDetIdItemList_[ic].items.begin(),vDetIdItemList_[ic].items.end());
    vdetId_.insert(vdetId_.end(),vDetIdItemList_[ic].items.size(),vDetIdItemList_[ic].detid);
  }

  // Vector of graphs in this request and DetNames which correspond to them
  std::vector<TGraphErrors*> VectorOfGraphs;
  std::vector<std::string> VectorOfDetNames;

  for(size_t i=0;i<vlistItems_.size();++i){
    std::cout <<  "TkRegion " << vdetId_[i] << " " << vlistItems_[i] << std::endl;

    if(vlistItems_.at(i).find("Summary")!= std::string::npos) vlistItems_.at(i).replace(vlistItems_.at(i).find("Summary_"),8,"");
    if(vlistItems_.at(i).find(fSep)!= std::string::npos) vlistItems_.at(i).replace(vlistItems_.at(i).find(fSep),fSep.size(),"_");
    
 
    std::stringstream ss;
    if (fHDQMInspectorConfig != 0x0) {
      ss << fHDQMInspectorConfig->translateDetId( vdetId_[i] ) << vlistItems_[i];
      VectorOfDetNames.push_back( fHDQMInspectorConfig->translateDetId( vdetId_[i] ));
    } else {
      ss << "Id " << vdetId_[i] << " " << vlistItems_[i];
      VectorOfDetNames.push_back( "???" );
    }

    
    bool const itemForIntegration = fHDQMInspectorConfig ? fHDQMInspectorConfig->computeIntegral(vlistItems_[i]) : false;
   
    int addShift=0;
    for(size_t j=0;j<vRun_.size();++j){
      index=j*vlistItems_.size()+i;
      X[j]=vRun_[j];
      EX[j]=0;
      Y[j]=vSummary_[index];
      //if (Y[j]==-10 || Y[j]==-9999 || Y[j] ==-99) {EY[j] = 0; Y[j] = 0;}
       
      // -9999 : existing HDQMSummary object in DB but part of the information not uploaded
      // -99   : HDQMSummary object not existing for this detId, informations are missing for all quantities 
      // -10 bad fit ?

      //std::cout << "dhidas " << vlistItems_[i] << "  " << vRun_[j] << "  " << vSummary_[index] << std::endl;
     
      if(vlistItems_[i].find("mean")!=std::string::npos){
        //if the quantity requested is mean, the error is evaluated as the error on the mean=rms/sqrt(entries)
        EY[j]=vSummary_[index+2]>0?vSummary_[index+1]/sqrt(vSummary_[index+2]):0;
        addShift=2;
      }else if (vlistItems_[i].find("entries")!=std::string::npos) {
        addShift=0;
      }else if (vlistItems_[i].find("landauPeak")!=std::string::npos){
        EY[j]=vSummary_[index+1];
        addShift=1;
      }
      else if (vlistItems_[i].find("gaussMean")!=std::string::npos){
        EY[j]=vSummary_[index+1];
        addShift=1;
      }
      else if (vlistItems_[i].find("Chi2NDF")!=std::string::npos || vlistItems_[i].find("rms")!=std::string::npos){
        EY[j]= 0.;
      }
      else {
        //EY[j]=vSummary_[index+1];
        EY[j]=0;// dhidas hack fix for now.  needs to be fixed
        addShift=1;
      }

      // integrate
      if (j == 0 ) YCumul[j] = Y[j]; 
      else         YCumul[j] = Y[j] + YCumul[j-1];

      // dhidas HACK for now
      EY[j] = 0;

      if(iDebug) {
        std::cout << index-j*vlistItems_.size() <<  " " << j  << " " << X[j]  << " " << Y[j] << " " << EY[j] << std::endl;
      }
    }

    C->cd(++padCount);
    gPad->SetLogy(logy);

    // Loop over all values and correct them for user defined range
    if (UseYRange != 0) {
      for (size_t iRun = 0; iRun != vRun_.size(); ++iRun) {
        if (UseYRange % 2 == 1 && Y[iRun] < YMin) {
          Y[iRun] = YMin;
          EY[iRun] = 0;
        }
        if (UseYRange >= 2  && Y[iRun] > YMax) {
          Y[iRun] = YMax;
          EY[iRun] = 0;
        }
      }
    }
    
    graph = new TGraphErrors((int) vRun_.size(),X,Y,EX,EY);
    if( fSkip99s || fSkip0s ) {
      int iptTGraph = 0;
      for (size_t ipt = 0; ipt != vRun_.size(); ++ipt) {
        // skip 99s or 0s when requested
        // std::cout << "point = " << Y[ipt] << std::endl;
        // if( Y[ipt] == 0 ) {
        //   std::cout << "fSkip0s = " << fSkip0s << std::endl;
        // }
        // if( (Y[ipt] == -10 || Y[ipt] == -9999 || Y[ipt] == -999 || Y[ipt] == -99) ) {
        //   std::cout << "fSkip99s = " << fSkip99s << std::endl;
        // }
        if( ((Y[ipt] == -10 || Y[ipt] == -9999 || Y[ipt] == -999 || Y[ipt] == -99) && fSkip99s) || (Y[ipt] == 0 && fSkip0s) ) {
          // std::cout << "removing point Y["<<ipt<<"] = " << Y[ipt] << ", when graph->GetN() = " << graph->GetN() << " and iptTGraph = " << iptTGraph << std::endl;
          // Int_t point = graph->RemovePoint(iptTGraph);
          // std::cout << "point removed = " << point << std::endl;
          graph->RemovePoint(iptTGraph);
        }
        else {
          // The TGraph is shrinked everytime a point is removed. We use another counter that
          // is increased only when not removing elements from the TGraph.
          ++iptTGraph;
        }
      }
    }
        
    graph->SetTitle(ss.str().c_str());
    if (UseYRange % 2 == 1) {
      graph->SetMinimum(YMin);
    }
    if (UseYRange >= 2) {
      graph->SetMaximum(YMax);
    }

    graph->Draw("Ap");
    graph->SetName(ss.str().c_str());
    graph->GetXaxis()->SetTitle("Run number");
    graph->Write();

    // put the graph in the vector eh.
    VectorOfGraphs.push_back(graph);

    // dhidas
    // Want to get some values into file... testing
    //for (int iDean = 0; iDean != graph.GetN(); ++iDean) {
    //  static std::ofstream OutFile("DeanOut.txt");
    //  fprintf("%9i %9i %12.3f\n", iDean, graph.GetX()[iDean], graph.GetY()[iDean]);
    //}

    if (itemForIntegration)
    {  
      std::stringstream ss2; std::stringstream ss3; std::stringstream ss4;
      std::string title =  vlistItems_.at(i);
     
      ss2 << title << "_Integral";
      ss3 << title << "_Integrated.gif";
      ss4 << title << "_Integrated.root";

      TCanvas* C2 = new TCanvas(ss2.str().c_str(),"");
      TGraphErrors* graph2 = new TGraphErrors((int) vRun_.size(),X,YCumul,EX,EX);
      graph2->SetTitle(ss2.str().c_str());
      graph2->SetMarkerColor(1);
      graph2->Draw("Ap");
      graph2->SetName(ss2.str().c_str());
      graph2->GetXaxis()->SetTitle("Run number");
      graph2->Write();
      C2->Write();
      C2->SaveAs(ss3.str().c_str());
      C2->SaveAs(ss4.str().c_str());
      // dhidas commented out below because it doesn't seem useful.  uncomment if you like, it was just annoying me.
      //C2->SaveAs(ss3.str().replace(ss3.str().find("."),ss3.str().size()-ss3.str().find("."),".C").c_str());
      }
    i+=addShift;
  }
  C->Write();
  C->SaveAs(CanvasName.c_str());
  // dhidas commented out below because it doesn't seem useful.  uncomment if you like, it was just annoying me.
  //C->SaveAs(CanvasName.replace(CanvasName.find("."),CanvasName.size()-CanvasName.find("."),".C").c_str());//savewith .C
  // dhidas commented out below because it doesn't seem useful.  uncomment if you like, it was just annoying me.
  //C->SaveAs(CanvasName.replace(CanvasName.find("."),CanvasName.size()-CanvasName.find("."),".C").c_str());//savewith .C


  // Okay, we wrote the first canvas, not let's try to overlay the graphs on another one..
  if (VectorOfGraphs.size() > 1) {

    // Create the legend for this overlay graph
    TLegend OverlayLegend(0.80,0.35,0.99,0.65);

    // Use for storing the global min/max.
    float max = -9999;
    float min =  9999;

    // Canvas we'll paint the overlays on
    TCanvas DeanCan("DeanCan", "DeanCan");
    TVirtualPad* VPad = DeanCan.cd();
    VPad->SetRightMargin(0.21);
    VPad->SetTopMargin(0.13);

    // Replace default legend names with labels if they exist
    TString const LNames = Labels;
    TObjArray* MyArrayPtr = LNames.Tokenize(",");
    if (MyArrayPtr) {
      MyArrayPtr->SetOwner(kTRUE);
      for( int i = 0; i <= MyArrayPtr->GetLast(); ++i ) {
        if( i < int(VectorOfDetNames.size()) ) {
          VectorOfDetNames[i] = ((TObjString*) MyArrayPtr->At(i) )->GetString().Data();
        }
      }
      MyArrayPtr->Delete();
    }
        

    // Let's loop over all graphs in this request
    for (size_t i = 0; i != VectorOfGraphs.size(); ++i) {

      // Strip off the det name in the i-th hist title
      TString MyTitle = VectorOfGraphs[i]->GetTitle();
      std::cout << "dhidas " << MyTitle << " : " << VectorOfDetNames[i] << std::endl;
      MyTitle.ReplaceAll(VectorOfDetNames[i]+"_", "");
      MyTitle.ReplaceAll("_"+VectorOfDetNames[i], "");
      MyTitle.ReplaceAll(VectorOfDetNames[i], "");
      std::cout << "dhidas " << MyTitle << std::endl;
      VectorOfGraphs[i]->SetTitle( MyTitle );

      // Add this to the legend, sure, good
      OverlayLegend.AddEntry(VectorOfGraphs[i], VectorOfDetNames[i].c_str(), "p");

      // You have to get the min and max by hand because root is completely retarded
      if (min > findGraphMin(VectorOfGraphs[i]) ) {
        min = findGraphMin(VectorOfGraphs[i]);
      }
      if (max < findGraphMax(VectorOfGraphs[i])) {
        max = findGraphMax(VectorOfGraphs[i]);
      }

      // let's use these colors and shapes for now
      VectorOfGraphs[i]->SetMarkerStyle(20+i);
      VectorOfGraphs[i]->SetMarkerColor(2+i);
    }
    // May as well set the min and max for first graph we'll draw
    VectorOfGraphs[0]->SetMinimum((min)-((max)-(min))/5.);
    VectorOfGraphs[0]->SetMaximum((max)+((max)-(min))/5.);
    if (UseYRange % 2 == 1) {
      VectorOfGraphs[0]->SetMinimum(YMin);
    }
    if (UseYRange >= 2) {
      VectorOfGraphs[0]->SetMaximum(YMax);
    }

    // Draw the first one with axis (A) and the rest just points (p), draw the legend, and save that canvas
    
    VectorOfGraphs[0]->Draw("Ap");
    for (size_t i = 1; i != VectorOfGraphs.size(); ++i) {
      VectorOfGraphs[i]->Draw("p");
    }
    OverlayLegend.Draw("same");
    //OverlayLegend.SetTextSize(1.5);
    DeanCan.SaveAs(CanvasName.replace(CanvasName.find("."),CanvasName.size()-CanvasName.find("."),"_Overlay.gif").c_str());
  }

  // While I'm here I may as well try deleting the graphs since people don't like to clean up after themselves
  for (size_t i = 0; i != VectorOfGraphs.size(); ++i) {
    delete VectorOfGraphs[i];
  }

  // Why do people never put a friggin return statement?
  return;

}

size_t HDQMInspector::unpackItems(std::string& ListItems)
{
  std::string::size_type oldloc=0; 
  std::string::size_type loc = ListItems.find( ",", oldloc );
  size_t count=1;
  while( loc != std::string::npos ) {
    setItems(ListItems.substr(oldloc,loc-oldloc));
    oldloc=loc+1;
    loc=ListItems.find( ",", oldloc );
    count++; 
  } 
  //there is a single item
  setItems(ListItems.substr(oldloc,loc-oldloc));
  std::cout << std::endl;
  return count;
}

void HDQMInspector::unpackConditions( std::string& Conditions, std::vector<DetIdItemList>& vdetIdItemList)
{
  char * pch;
  char delimiters[128]="><=+-*/&|() ";
  char copyConditions[1024];
  sprintf(copyConditions,"%s",Conditions.c_str());
  pch = strtok (copyConditions,delimiters);
  while (pch != NULL){
    if(strstr(pch,fSep.c_str())!=NULL){
      DetIdItemList detiditemlist;
      std::string itemD(pch);
      detiditemlist.detid=atol(itemD.substr(0,itemD.find(fSep)).c_str());
      detiditemlist.items.push_back(itemD.substr(itemD.find(fSep)+fSep.size())); // dhidas update +.size instead of "1"
      if (iDebug) {
        std::cout << "Found a Condition " << detiditemlist.items.back() << " for detId " << detiditemlist.detid << std::endl;
      }
      
      if(vdetIdItemList.size())
        if(vdetIdItemList.back().detid==detiditemlist.detid)
          vdetIdItemList.back().items.insert(vdetIdItemList.back().items.end(),detiditemlist.items.begin(),detiditemlist.items.end());
        else
          vdetIdItemList.push_back(detiditemlist);
      else
        vdetIdItemList.push_back(detiditemlist); 
    }
    pch = strtok (NULL,delimiters);
  }
}

bool HDQMInspector::ApplyConditions(std::string& Conditions, std::vector<DetIdItemList>& vdetIdItemList)
{
  double resultdbl=1;
  char cConditions[1024];
  char singleCondition[1024];
  char condCVal[1024];   

  sprintf(cConditions,"%s",Conditions.c_str());
  if (iDebug) {
    std::cout << "Conditions " << cConditions << std::endl;
  }
  for(size_t ic=0;ic<vdetIdItemList.size();++ic)
    for(size_t jc=0;jc<vdetIdItemList[ic].items.size();++jc){
      //scientific precision doesn't work in HDQMExpressionEvaluator...
      //sprintf(condCVal,"%g",vdetIdItemList[ic].values[jc]);
      sprintf(condCVal,"%f",vdetIdItemList[ic].values[jc]);
      sprintf(singleCondition,"%d%s%s",vdetIdItemList[ic].detid,fSep.c_str(),vdetIdItemList[ic].items[jc].c_str());
      //printf("dhidas %d  %s  %s\n",vdetIdItemList[ic].detid,fSep.c_str(),vdetIdItemList[ic].items[jc].c_str());
      //printf("dhidas %s %s\n", cConditions, singleCondition);
      char* fpos = strstr(cConditions,singleCondition);
      //printf("dhidas %s %s %i\n", fpos, condCVal, strlen(condCVal));
      strncpy(fpos,condCVal,strlen(condCVal));
      memset(fpos+strlen(condCVal),' ',strlen(singleCondition)-strlen(condCVal));
      //std::cout << "fpos " << fpos << " len condCVal " << strlen(condCVal) << " strlen(singleCondition) " << strlen(singleCondition) << " len cConditions " << strlen(cConditions)<<std::endl;
      //std::cout << "Conditions Replace: Condition " << singleCondition << " string changed in " << cConditions << std::endl;
    }

  std::string stringToEvaluate;
  char * pch;
  pch = strtok (cConditions," ");
  while (pch != NULL){
    stringToEvaluate.append(pch);
    pch = strtok (NULL, " ");
  } 
  //for(size_t i=0;i<strlen(cConditions);++i)
  // if(cConditions[i] != " ")
  //  stringToEvaluate.push_back(cConditions[i]);

  if(iDebug) {
    std::cout << "Conditions After SubStitution " << stringToEvaluate << std::endl;
  }
  TFormula Formula("condition", stringToEvaluate.c_str());
  resultdbl = Formula.Eval(0);
  if(iDebug) {
    std::cout << "Result " << resultdbl << std::endl;
  }
  if(!resultdbl) {
    return false;
  }
  return true;
}

void HDQMInspector::setItems(std::string itemD)
{
  DetIdItemList detiditemlist;
  detiditemlist.detid=atol(itemD.substr(0,itemD.find(fSep)).c_str());

  std::string item=itemD.substr(itemD.find(fSep)+fSep.size());
  detiditemlist.items.push_back(item);

  if(iDebug)
    std::cout << "Found new item " << detiditemlist.items.back() << " for detid " << detiditemlist.detid << std::endl;

  if(item.find("mean")!=std::string::npos){
    detiditemlist.items.push_back(item.replace(item.find("mean"),4,"rms")); 
    if(iDebug)
      std::cout << "Found new item " << detiditemlist.items.back() << std::endl;
    detiditemlist.items.push_back(item.replace(item.find("rms"),3,"entries")); 
    if(iDebug)
      std::cout << "Found new item " << detiditemlist.items.back() << std::endl;
  }
  else if(item.find("landauPeak")!=std::string::npos){
    detiditemlist.items.push_back(item.replace(item.find("landauPeak"),10,"landauPeakErr")); 
    if(iDebug)
      std::cout << "Found new item " << detiditemlist.items.back() << std::endl;
  }
  else if(item.find("gaussMean")!=std::string::npos){
    detiditemlist.items.push_back(item.replace(item.find("gaussMean"),9,"gaussSigma")); 
    if(iDebug)
      std::cout << "Found new item " << detiditemlist.items.back() << std::endl;
  }

  if(vDetIdItemList_.size()) {
    if(vDetIdItemList_.back().detid==detiditemlist.detid) {
      vDetIdItemList_.back().items.insert(vDetIdItemList_.back().items.end(),detiditemlist.items.begin(),detiditemlist.items.end());
    } else {
      vDetIdItemList_.push_back(detiditemlist);
    }
  } else {
    vDetIdItemList_.push_back(detiditemlist);
  }

  return;
}


double HDQMInspector::findGraphMax(TGraphErrors* g)
{
  // GetMaximum() doesn't work ....
  int n = g->GetN();
  double* y = g->GetY();
  int locmax = TMath::LocMax(n,y);
  assert(y != 0);
  return  y[locmax];
}


double HDQMInspector::findGraphMin(TGraphErrors* g)
{
  // GetMinimum() doesn't work ....
  int n = g->GetN();
  double* y = g->GetY();
  int locmin = TMath::LocMin(n,y);
  assert(y != 0);
  return  y[locmin];
}


