/*
 * Make some convenient Ecal function available in python
 * 
 * \author Stefano Argiro
 * \version $Id: EcalPyUtils.cc,v 1.7 2012/07/18 10:40:33 davidlt Exp $
 */


#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondTools/Ecal/interface/EcalFloatCondObjectContainerXMLTranslator.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;



namespace ecalpyutils{
  
  std::vector<int> hashedIndexToEtaPhi(int hashedindex){
    int ieta= EBDetId::unhashIndex(hashedindex).ieta();
    int iphi= EBDetId::unhashIndex(hashedindex).iphi();
    std::vector<int> ret;
    ret.push_back(ieta);
    ret.push_back(iphi);
    return ret;
  }
  
  std::vector<int> hashedIndexToXY(int hashedindex){
    int ix= EEDetId::unhashIndex(hashedindex).ix();
    int iy= EEDetId::unhashIndex(hashedindex).iy();
    int zside =  EEDetId::unhashIndex(hashedindex).zside();
    std::vector<int> ret;
    ret.push_back(ix);
    ret.push_back(iy);
    ret.push_back(zside);
    return ret;
  }

  
  int hashedIndex(int ieta, int iphi){
    EBDetId d(ieta,iphi);
    return d.hashedIndex();
  }

  int hashedIndexEE(int ix, int iy, int iz){

    if (EEDetId::validDetId(ix,iy,iz)) {
      EEDetId d(ix,iy,iz);
      return d.hashedIndex();
    }
    return 0;
  }

  int ism(int ieta, int iphi){
    EBDetId d(ieta,iphi);
    return d.ism();
  }

  std::string arraystoXML(const std::vector<float>& eb, const std::vector<float>& ee){
    EcalCondHeader h;
    return EcalFloatCondObjectContainerXMLTranslator::dumpXML(h,eb,ee);
  }
}

BOOST_PYTHON_MODULE(pluginEcalPyUtils) {

//   looks like these are already defined somewhere

//   python access to stl integer vectors 
//   class_< std::vector<int> >("vectorInt") 
//     .def(vector_indexing_suite<std::vector<int> >()) 
//   ; 

//   class_< std::vector<float> >("vectorFloat") 
//     .def(vector_indexing_suite<std::vector<float> >()) 
//   ; 

  def("hashedIndexToEtaPhi",&ecalpyutils::hashedIndexToEtaPhi);
  def("hashedIndexToXY",&ecalpyutils::hashedIndexToXY);
  def("hashedIndex",&ecalpyutils::hashedIndex);
  def("hashedIndexEE",&ecalpyutils::hashedIndexEE);
  def("ism",&ecalpyutils::ism);
  def("barrelfromXML",&EcalFloatCondObjectContainerXMLTranslator::barrelfromXML);
  def("endcapfromXML",&EcalFloatCondObjectContainerXMLTranslator::endcapfromXML);
  def("arraystoXML",&ecalpyutils::arraystoXML);
}

