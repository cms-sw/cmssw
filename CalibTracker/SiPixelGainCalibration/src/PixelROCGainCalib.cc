#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalib.h"
#include "CalibTracker/SiPixelGainCalibration/interface/PixelROCGainCalibPixel.h"

//PixelROCGainCalib::PixelROCGainCalib():nentries_(0),vcalrangestep_(16),vcalrangemin_(0),vcalrangemax_(256),nvcal_(0),ncolslimit_(300),nrowslimit_(300),thisROCTitle_("empty"),detid_(0)
//{
//
//  thePixels_.reserve(nrowslimit_);
//  for(uint32_t irow=0; irow<nrowslimit_; irow++)
//    thePixels_[irow].reserve(ncolslimit_);
//}



PixelROCGainCalib::PixelROCGainCalib(uint32_t maxrow, uint32_t maxcol,uint32_t nvcal) :
  thePixels_(maxrow,std::vector<PixelROCGainCalibPixel>(maxcol,PixelROCGainCalibPixel(nvcal))),
  nentries_(0),vcalrangestep_(16),vcalrangemin_(0),vcalrangemax_(256),nvcal_(nvcal),ncolslimit_(maxcol),nrowslimit_(maxrow),thisROCTitle_("empty"),detid_(0)
{
}
PixelROCGainCalib::~PixelROCGainCalib(){
  for(uint32_t irow=0; irow<nrowslimit_; irow++)
    thePixels_[irow].clear();
  thePixels_.clear();
}

void PixelROCGainCalib::fill(uint32_t row,uint32_t col,uint32_t ipoint,uint32_t adc, bool verbose){
  // this is where the filling should happen
  
  if(ipoint<points.size()){
    fillVcal(row,col,getBinVcal(ipoint),adc,verbose);
  }
}
void PixelROCGainCalib::fillVcal(uint32_t row,uint32_t col,uint32_t vcal,uint32_t adc, bool verbose){
  // this is where the filling should happen
  thePixels_[row][col].addPoint(getVcalBin(vcal),adc);
  nentries_++;
  //  if(verbose){
  if(1){
    edm::LogInfo("INFO") << "filling DetID " << detid_ << " row : " << row << ", col " << col << " vcal " << vcal << " with value " << adc << std::endl;
  }
 
  //  TH1F *hist = thePixels_[row][col];
//   if(hist){
//     hist->Fill(vcal,result);
//     if(0){// makes things very loud
//       std::cout << "filling DetID " << detid_ << " row : " << row << ", col " << col << " vcal " << vcal << " with value " << adc << ", histogram " << hist->GetName() << " now has " << hist->GetEntries() << " entries." << std::endl;
//       printsummary(row,col);
//     }
//     nentries_++;
//   }
  
}

TH1F *PixelROCGainCalib::gethisto(uint32_t row, uint32_t col, edm::Service < TFileService > therootfileservice){
  TString thehistname = createTitle(row,col);
  
  TH1F *hist = (therootfileservice->mkdir(thisROCTitle_.c_str())).make<TH1F>(thehistname.Data(),thehistname.Data(),nvcal_+1,vcalrangemin_-(0.5*vcalrangestep_),vcalrangemax_+(0.5*vcalrangestep_));
  //  std::cout << "created histogram with name \"" << thehistname << "\"" << std::endl;
  for(uint32_t ipoint=0;ipoint<nvcal_; ipoint++){
    //    std::cout <<" -- bin " << ipoint << " " <<  getBinVcal(ipoint) << " " << thePixels_[row][col].getpoint(ipoint,ntriggers_) << std::endl;
    hist->Fill(getBinVcal(ipoint),thePixels_[row][col].getpoint(ipoint,ntriggers_));
  }
  return hist;
	       
}

void PixelROCGainCalib::printsummary(uint32_t row, uint32_t col){
  
 edm::LogInfo("INFO") << "summary: created PixelROCGainCalib object with settings: " << std::endl;
    edm::LogInfo("INFO") << "name " << thisROCTitle_ << std::endl;
    edm::LogInfo("INFO") << "nvcal " << nvcal_ << std::endl;
    edm::LogInfo("INFO") << "vcalrangestep " << vcalrangestep_ << std::endl;
    edm::LogInfo("INFO") << "vcalrangemin " << vcalrangemin_ << std::endl;
    edm::LogInfo("INFO") << "vcalrangemax " << vcalrangemax_ << std::endl;
    edm::LogInfo("INFO") << "ncolums " << ncolslimit_ << std::endl;
    edm::LogInfo("INFO") << "nrows " << nrowslimit_ << std::endl;
    edm::LogInfo("INFO") << "looking at pixel at ROW: " << row << ", COLUMN: " << col << std::endl;
    //    TH1F * hist = thePixels_[row][col]->GetHist();
    //    if(hist){
    //      for(int ibin=0; ibin<hist->GetNbinsX() ;ibin++)
    //	if(hist->GetBinContent(ibin)!=0)
    //	  edm::LogInfo("INFO") << "--- bin " << ibin <<", center "  << hist->GetBinCenter(ibin) << ", value " << hist->GetBinContent(ibin) << std::endl;
    //    }
    for(uint32_t ibin=0; ibin<nvcal_; ibin++){
      edm::LogInfo("INFO") << "--- bin " << ibin <<", center "  << getBinVcal(ibin) << ", value " << thePixels_[row][col].getpoint(ibin,ntriggers_) << std::endl;
    }
}

std::string PixelROCGainCalib::createTitle(uint32_t row, uint32_t col){
  TString result = "";
  result+=thisROCTitle_.c_str();
  result+=", row ";
  result+=row;
  result+=", col ";
  result+=col;
  std::string res;
  res+=result.Data();
  return res;
}
void PixelROCGainCalib::init(std::string name, uint32_t detid,uint32_t nvcal,uint32_t vcalRangeMin, uint32_t vcalRangeMax, unsigned vcalRangeStep,uint32_t ncols, uint32_t nrows, uint32_t ntriggers, edm::Service<TFileService>  therootfileservice){
  thisROCTitle_=name;
  detid_=detid;
  nvcal_=0;
  ntriggers_=ntriggers;
  vcalrangemin_=vcalRangeMin;
  ncolslimit_=ncols;
  nrowslimit_=nrows;
  nvcal_=nvcal;
  vcalrangestep_=vcalRangeStep;
  vcalrangemax_=vcalRangeMax;
  vcalrangemin_=vcalRangeMin;
  for(uint32_t i=0; i<nvcal_; i++){
    points.push_back(vcalrangemin_+(i*vcalrangestep_));
    points[i]=vcalrangemin_+(i*vcalrangestep_);
  }
  // and fill the vector with pixel objects
  TString tempstring = "detid_";
  tempstring+=detid_;
  std::string tempstr = tempstring.Data();
  TFileDirectory thisdir = therootfileservice->mkdir(tempstr,tempstr);
 
  for(uint32_t irow=0; irow<nrowslimit_; irow++){
    std::vector < PixelROCGainCalibPixel > thevec;
    for(uint32_t icol=0; icol<ncolslimit_; icol++){
      //  TString thehistname = "";
      //      thehistname += getTitle();
      //      thehistname += ", row ";
      //      thehistname += irow;
      //      thehistname += ", col ";
      //      thehistname += icol;
      //      TH1F *pix = thisdir.make<TH1F>(thehistname.Data(),thehistname.Data(),nvcal_,vcalrangemin_-(0.5*vcalrangestep_), vcalrangemax_+(0.5*vcalrangestep_));
      PixelROCGainCalibPixel histo(nvcal_);
      thevec.push_back( histo );
    }
    thePixels_.push_back(thevec);
  }
  for(uint32_t irow=0; irow<nrowslimit_; irow++){
    for(uint32_t icol=0; icol<ncolslimit_; icol++){
      thePixels_[irow][icol].init(nvcal_);
    }
  }
 
}

bool PixelROCGainCalib::isfilled(uint32_t row,uint32_t col){
  return thePixels_[row][col].isfilled();
}
