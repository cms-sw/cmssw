#include "Alignment/OfflineValidation/interface/PVValidationHelpers.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>
//#include "TH1.h"

//*************************************************************
void PVValHelper::add(std::map<std::string, TH1*>& h, TH1* hist)
//*************************************************************
{ 
  h[hist->GetName()]=hist; 
  //hist->StatOverflows(kTRUE);
}

//*************************************************************
void PVValHelper::fill(std::map<std::string, TH1*>& h,const std::string& s, double x)
//*************************************************************
{
  if(h.count(s)==0){
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x);
}

//*************************************************************
void PVValHelper::fill(std::map<std::string, TH1*>& h,const std::string& s, double x, double y)
//*************************************************************
{
  if(h.count(s)==0){
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram named " << s << std::endl;
    return;
  }
  h[s]->Fill(x,y);
}

//*************************************************************
void PVValHelper::fillByIndex(std::vector<TH1F*>& h, unsigned int index, double x,std::string tag)
//*************************************************************
{
  assert(!h.empty());
  if(index <= h.size()){
    h[index]->Fill(x);
  } else {
    edm::LogWarning("PVValidationHelpers") << "Trying to fill non-existing Histogram with index " << index << " for array with size: "<<h.size()<<" tag: "<< tag<< std::endl;
    return;
  }
}

//*************************************************************
void PVValHelper::shrinkHistVectorToFit(std::vector<TH1F*>&h, unsigned int desired_size)
//*************************************************************
{
  h.erase(h.begin()+desired_size,h.end()); 
}

//*************************************************************
std::tuple<std::string,std::string,std::string> PVValHelper::getTypeString (PVValHelper::residualType type)
//*************************************************************
{
  std::tuple<std::string,std::string,std::string> returnType;
  switch(type)
    {
    // absoulte

    case PVValHelper::dxy  :
      returnType = std::make_tuple("dxy","d_{xy}","[#mum]");
      break;
    case PVValHelper::dx   :
      returnType = std::make_tuple("dx","d_{x}","[#mum]");
      break;
    case PVValHelper::dy   :
      returnType = std::make_tuple("dy","d_{y}","[#mum]");
      break;
    case PVValHelper::dz   :
      returnType =  std::make_tuple("dz","d_{z}","[#mum]");
      break;
    case PVValHelper::IP2D :
      returnType =  std::make_tuple("IP2D","IP_{2D}","[#mum]");
      break;
    case PVValHelper::resz :
      returnType =  std::make_tuple("resz","z_{trk}-z_{vtx}","[#mum]");
      break;
    case PVValHelper::IP3D :
      returnType =  std::make_tuple("IP3D","IP_{3D}","[#mum]");
      break;
    case PVValHelper::d3D  : 
      returnType =  std::make_tuple("d3D","d_{3D}","[#mum]");
      break;

    // normalized

    case PVValHelper::norm_dxy  :
      returnType =  std::make_tuple("norm_dxy","d_{xy}/#sigma_{d_{xy}}","");
      break;
    case PVValHelper::norm_dx   :
      returnType =  std::make_tuple("norm_dx","d_{x}/#sigma_{d_{x}}","");
      break;
    case PVValHelper::norm_dy   :
      returnType =  std::make_tuple("norm_dy","d_{y}/#sigma_{d_{y}}","");
      break;
    case PVValHelper::norm_dz   :
      returnType =  std::make_tuple("norm_dz","d_{z}/#sigma_{d_{z}}","");
      break;
    case PVValHelper::norm_IP2D :
      returnType =  std::make_tuple("norm_IP2D","IP_{2D}/#sigma_{IP_{2D}}","");
      break;
    case PVValHelper::norm_resz :
      returnType =  std::make_tuple("norm_resz","z_{trk}-z_{vtx}/#sigma_{res_{z}}","");
      break;
    case PVValHelper::norm_IP3D :
      returnType =  std::make_tuple("norm_IP3D","IP_{3D}/#sigma_{IP_{3D}}","");
      break;
    case PVValHelper::norm_d3D  : 
      returnType =  std::make_tuple("norm_d3D","d_{3D}/#sigma_{d_{3D}}","");
      break;

    default:
      edm::LogWarning("PVValidationHelpers") <<" getTypeString() unknown residual type: "<<type<<std::endl;
    }

  return returnType;
  
}

//*************************************************************
std::tuple<std::string,std::string,std::string> PVValHelper::getVarString (PVValHelper::plotVariable var)
//*************************************************************
{
  std::tuple<std::string,std::string,std::string> returnVar;
  switch(var)
    {
    case PVValHelper::phi  :
      returnVar =  std::make_tuple("phi","#phi","[rad]");
      break;
    case PVValHelper::eta   :
      returnVar =  std::make_tuple("eta","#eta","");
      break;
    case PVValHelper::pT   :
      returnVar = std::make_tuple("pT","p_{T}","[GeV]");
      break;
    case PVValHelper::pTCentral :
      returnVar = std::make_tuple("pTCentral","p_{T} |#eta|<1.","[GeV]");
      break;
    case PVValHelper::ladder   :
      returnVar = std::make_tuple("ladder","ladder number","");
      break;
    case PVValHelper::modZ :
      returnVar = std::make_tuple("modZ","module number","");
      break;
    default:
      edm::LogWarning("PVValidationHelpers") <<" getVarString() unknown plot variable: "<<var<<std::endl;
    }

  return returnVar;
  
}
