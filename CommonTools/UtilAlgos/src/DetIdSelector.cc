#include "CommonTools/UtilAlgos/interface/DetIdSelector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


DetIdSelector::DetIdSelector():
  m_selections(),m_masks()
{}

DetIdSelector::DetIdSelector(const std::string& selstring):
  m_selections(),m_masks()
{
  addSelection(selstring);
}

DetIdSelector::DetIdSelector(const std::string& selstring, const std::string label):
  m_selections(),m_masks()
{
  if (label == "bits")   addSelection(selstring);
  else if (label == "words")  addSelectionByWords(selstring);
  else edm::LogError("Selection added") << "DetIdSelector construct with label, but label not valid!";
}

DetIdSelector::DetIdSelector(const std::vector<std::string>& selstrings):
  m_selections(),m_masks()
{
  addSelection(selstrings);
}

DetIdSelector::DetIdSelector(const std::vector<std::string>& selstrings, const std::string label):
  m_selections(),m_masks()
{
  if (label == "bits")   addSelection(selstrings);
  else if (label == "words")  addSelectionByWords(selstrings);
  else edm::LogError("Selection added") << "DetIdSelector construct with label, but label not valid!";
}

DetIdSelector::DetIdSelector(const edm::ParameterSet& selconfig):
  m_selections(),m_masks()
{

  const std::vector<std::string> selstrings = selconfig.getUntrackedParameter<std::vector<std::string> >("selection");
  addSelection(selstrings);

}

void DetIdSelector::addSelection(const std::string& selstring) {

  unsigned int selection;
  unsigned int mask;

  if(selstring.substr(0,2) == "0x") {
    sscanf(selstring.c_str(),"%x-%x",&mask,&selection);
  }
  else {
    sscanf(selstring.c_str(),"%u-%u",&mask,&selection);
  }

  m_selections.push_back(selection);
  m_masks.push_back(mask);

  LogDebug("Selection added") << "Selection " << selection << " with mask " << mask << " added";

}

void DetIdSelector::addSelection(const std::vector<std::string>& selstrings) {

  for(std::vector<std::string>::const_iterator selstring=selstrings.begin();selstring!=selstrings.end();++selstring) {
    addSelection(*selstring);
  }

}

void DetIdSelector::addSelectionByWords(const std::string& selstring) {
  //LogTrace("OccupancyPlots") << "m_labels.push_back: " << selstring;

  m_labels.push_back(selstring);

}

void DetIdSelector::addSelectionByWords(const std::vector<std::string>& selstrings) {

  for(std::vector<std::string>::const_iterator selstring=selstrings.begin();selstring!=selstrings.end();++selstring) {
    addSelectionByWords(*selstring);
  }

}

bool DetIdSelector::isSelected(const unsigned int& rawid) const {

  for(unsigned int i=0; i<m_selections.size() ; ++i) {
    if((m_masks[i] & rawid) == m_selections[i]) return true;
  }

  return false;
}

bool DetIdSelector::isSelected(const DetId& detid) const {

  return isSelected(detid.rawId());

}


bool DetIdSelector::isSelectedByWords(const unsigned int& rawid, const TrackerTopology* tTopo) const {

  DetId detId = DetId(rawid);
  return isSelectedByWords(detId, tTopo);

}


bool DetIdSelector::isSelectedByWords(const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  for(unsigned int i=0; i<m_labels.size() ; ++i) {

    //LogTrace("OccupancyPlots") << "string: " << m_labels[i];

    pos = m_labels[i].find("PXB");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is PXB!!";
      isSelected = isSelectedByWordsPXB( m_labels[i], detid, tTopo);
    }

    pos = m_labels[i].find("PXF");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is PXF!!";
      isSelected = isSelectedByWordsPXF( m_labels[i], detid, tTopo);
    }

    pos = m_labels[i].find("TIB");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is TIB!!";
      isSelected = isSelectedByWordsTIB( m_labels[i], detid, tTopo);
    }

    pos = m_labels[i].find("TOB");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is TOB!!";
      isSelected = isSelectedByWordsTOB( m_labels[i], detid, tTopo);
    }

    pos = m_labels[i].find("TID");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is TID!!";
      isSelected = isSelectedByWordsTID( m_labels[i], detid, tTopo);
    }

    pos = m_labels[i].find("TEC");
    if(pos>=0){
      //LogTrace("OccupancyPlots") << "the label is TEC!!";
      isSelected = isSelectedByWordsTEC( m_labels[i], detid, tTopo);
    }
  }

//  uint32_t subdet = detid.subdetId();
//  for(unsigned int i=0; i<m_labels.size() ; ++i) {
//    if( m_labels[i].find("PXB") && subdet == PixelSubdetector::PixelBarrel ) return isSelectedByWords(detid.rawId(), tTopo);
//  }

  return isSelected;

}

bool DetIdSelector::isSelectedByWordsPXB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == PixelSubdetector::PixelBarrel){		//PXB_
    //LogTrace("OccupancyPlots") << "the module is PXB!!";
    isSelected = true;
  } else {
    return false;
  }

  pos = label.find("Layer");				//LayerXX_
  //LogTrace("OccupancyPlots") << "pos Layer:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains layer!!";
    isSelected = isSame(label, "Layer", tTopo->pxbLayer(detid.rawId()), 2);
    if (!isSelected) return false;
  } 

  pos = label.find("Ladder");				//LadderXX_
  //LogTrace("OccupancyPlots") << "pos Ladder:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains ladder!!";
    isSelected = isSame(label, "Ladder", tTopo->pxbLadder(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Module");				//ModuleXXX_
  //LogTrace("OccupancyPlots") << "pos Module:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->pxbModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }

  return isSelected;
}

bool DetIdSelector::isSelectedByWordsPXF(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == PixelSubdetector::PixelEndcap){		//PXB_
    //LogTrace("OccupancyPlots") << "the module is PXF!!";
    isSelected = true;
  } else {
    return false;
  }

  pos = label.find("Side");                            //SideX_
  //LogTrace("OccupancyPlots") << "pos Side:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Side!!";
    isSelected = isSame(label, "Side", tTopo->pxfSide(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Disk");                            //DiskXX_
  //LogTrace("OccupancyPlots") << "pos Disk:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Disk!!";
    isSelected = isSame(label, "Disk", tTopo->pxfDisk(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Blade");                            //BladeXX_
  //LogTrace("OccupancyPlots") << "pos Blade:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Blade!!";
    isSelected = isSame(label, "Blade", tTopo->pxfBlade(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Panel");                            //PanelX_
  //LogTrace("OccupancyPlots") << "pos Panel:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Panel!!";
    isSelected = isSame(label, "Panel", tTopo->pxfPanel(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Module");                           //ModuleXXX_
  //LogTrace("OccupancyPlots") << "pos Module:" << pos;
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->pxfModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }

  return isSelected;
}

bool DetIdSelector::isSelectedByWordsTIB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == StripSubdetector::TIB){          //TIB_
    //LogTrace("OccupancyPlots") << "the module is TIB!!";
    isSelected = true;
  } else {
    return false;
  }

  pos = label.find("Layer");                            //LayerXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains layer!!";
    isSelected = isSame(label, "Layer", tTopo->tibLayer(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Side");                            //SideX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Side!!";
    isSelected = isSame(label, "Side", tTopo->tibSide(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Order");                           //OrderX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Order!!";
    isSelected = isSame(label, "Order", tTopo->tibOrder(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Module");                           //ModuleXXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->tibModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }
  pos = label.find("Stereo");                           //Stereo
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Stereo!!";
    if(tTopo->isStereo(detid)) {
      //LogTrace("OccupancyPlots") << "the module is Stereo!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("RPhi");                           //RPhi
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains RPhi!!";
    if(tTopo->isRPhi(detid)) {
      //LogTrace("OccupancyPlots") << "the module is RPhi!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Upper");                           //Upper
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Upper!!";
    if(tTopo->isUpper(detid)){
      //LogTrace("OccupancyPlots") << "the module is Upper!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Lower");                           //Lower
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Lower!!";
    if(tTopo->isLower(detid)){
      //LogTrace("OccupancyPlots") << "the module is Lower!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  return isSelected;

}

bool DetIdSelector::isSelectedByWordsTOB(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == StripSubdetector::TOB){          //TOB_
    //LogTrace("OccupancyPlots") << "the module is TOB!!";
    isSelected = true;
  } else {
    return false;
  }


  pos = label.find("Layer");                            //LayerXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains layer!!";
    isSelected = isSame(label, "Layer", tTopo->tobLayer(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Rod");                           //RodXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains ladder!!";
    isSelected = isSame(label, "Rod", tTopo->tobRod(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Module");                           //ModuleXXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->tobModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }

  pos = label.find("Stereo");                           //Stereo
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Stereo!!";
    if(tTopo->isStereo(detid)) {
      //LogTrace("OccupancyPlots") << "the module is Stereo!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("RPhi");                           //RPhi
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains RPhi!!";
    if(tTopo->isRPhi(detid)) {
      //LogTrace("OccupancyPlots") << "the module is RPhi!";
      isSelected = true;
    } else { 
      return false;
    }
  }

  pos = label.find("Upper");                           //Upper
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Upper!!";
    if(tTopo->isUpper(detid)){
      //LogTrace("OccupancyPlots") << "the module is Upper!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Lower");                           //Lower
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Lower!!";
    if(tTopo->isLower(detid)){
      //LogTrace("OccupancyPlots") << "the module is Lower!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  return isSelected;

}

bool DetIdSelector::isSelectedByWordsTID(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == StripSubdetector::TID){          //TID_
    //LogTrace("OccupancyPlots") << "the module is TID!!";
    isSelected = true;
  } else {
    return false;
  }

  pos = label.find("Side");                            //SideX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Side!!";
    isSelected = isSame(label, "Side", tTopo->tidSide(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Wheel");                            //WheelXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Wheel!!";
    isSelected = isSame(label, "Wheel", tTopo->tidWheel(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Order");                            //OrderX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Order!!";
    isSelected = isSame(label, "Side", tTopo->tidOrder(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Ring");                           //RingXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Ring!!";
    isSelected = isSame(label, "Ring", tTopo->tidRing(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Module");                           //ModuleXXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->tidModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }

  pos = label.find("Stereo");                           //Stereo
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Stereo!!";
    if(tTopo->isStereo(detid)) {
      //LogTrace("OccupancyPlots") << "the module is Stereo!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("RPhi");                           //RPhi
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains RPhi!!";
    if(tTopo->isRPhi(detid)) {
      //LogTrace("OccupancyPlots") << "the module is RPhi!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Upper");                           //Upper
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Upper!!";
    if(tTopo->isUpper(detid)){
      //LogTrace("OccupancyPlots") << "the module is Upper!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Lower");                           //Lower
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Lower!!";
    if(tTopo->isLower(detid)){
      //LogTrace("OccupancyPlots") << "the module is Lower!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  return isSelected;

}

bool DetIdSelector::isSelectedByWordsTEC(std::string label, const DetId& detid, const TrackerTopology* tTopo) const {

  int pos = -1;
  bool isSelected = false;
  uint32_t subdet = detid.subdetId();
  if(subdet == StripSubdetector::TEC){          //TEC_
    //LogTrace("OccupancyPlots") << "the module is TEC!!";
    isSelected = true;
  } else {
    return false;
  }

  pos = label.find("Side");                            //SideX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Side!!";
    isSelected = isSame(label, "Side", tTopo->tecSide(detid.rawId()), 1);
    if (!isSelected) return false;
  }

  pos = label.find("Wheel");                            //WheelXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Wheel!!";
    isSelected = isSame(label, "Wheel", tTopo->tecWheel(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Order");                            //OrderX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Order!!";
    isSelected = isSame(label, "Side", tTopo->tecOrder(detid.rawId()), 1);
    if (!isSelected) return false;
  }


  pos = label.find("Ring");                            //RingXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains layer!!";
    isSelected = isSame(label, "Ring", tTopo->tecRing(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Petal");                           //PetalXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains ladder!!";
    isSelected = isSame(label, "Petal", tTopo->tecPetalNumber(detid.rawId()), 2);
    if (!isSelected) return false;
  }

  pos = label.find("Module");                           //ModuleXXX_
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Module!!";
    isSelected = isSame(label, "Module", tTopo->tecModule(detid.rawId()), 3);
    if (!isSelected) return false;
  }

  pos = label.find("Stereo");                           //Stereo
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Stereo!!";
    if(tTopo->isStereo(detid)) {
      //LogTrace("OccupancyPlots") << "the module is Stereo!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("RPhi");                           //RPhi
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains RPhi!!";
    if(tTopo->isRPhi(detid)) {
      //LogTrace("OccupancyPlots") << "the module is RPhi!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Upper");                           //Upper
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Upper!!";
    if(tTopo->isUpper(detid)){
      //LogTrace("OccupancyPlots") << "the module is Upper!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  pos = label.find("Lower");                           //Lower
  if(pos>=0){
    //LogTrace("OccupancyPlots") << "the label contains Lower!!";
    if(tTopo->isLower(detid)){
      //LogTrace("OccupancyPlots") << "the module is Lower!!";
      isSelected = true;
    } else {
      return false;
    }
  }

  return isSelected;

}

bool DetIdSelector::isSame(std::string label, std::string selection, unsigned int comparison, unsigned int spaces) const{

  int n = selection.length();
  int pos = -1;
  pos = label.find(selection);

  //check if one range exists after the selection
  std::string rangeSelected = label.substr(pos+n, 2*spaces+1);
  int pos2 = -1;
  pos2 = rangeSelected.find("-");
  if( pos2 > 0 ) {
    //LogTrace("OccupancyPlots") << "check if the range exists after the selection " << label.substr(pos+n, 2*spaces+1);
    return isInRange(rangeSelected, comparison, spaces);
  }
  unsigned int selected = std::stoi(label.substr(pos+n,spaces));
  //LogTrace("OccupancyPlots") << " selection from label " << label.substr(pos+n,spaces);
  //LogTrace("OccupancyPlots") << " selection from label " << selected;
  //LogTrace("OccupancyPlots") << " comparison from label " << comparison;
  if( selected == comparison) return true;
  return false;

}

bool DetIdSelector::isInRange(std::string range, unsigned int comparison, unsigned int spaces) const{

  //LogTrace("OccupancyPlots") << "RANGEEEEEEEE >> " << range;

  unsigned int rangeLower = std::stoi(range.substr(0,spaces));
  unsigned int rangeUpper = std::stoi(range.substr(spaces+1,spaces));
  //LogTrace("OccupancyPlots") << "RANGEEEEEEEE- >> " << rangeLower;
  //LogTrace("OccupancyPlots") << "RANGEEEEEEEE+ >> " << rangeUpper;
  if(comparison >= rangeLower && comparison <= rangeUpper) return true;
  return false;
}

bool DetIdSelector::operator()(const DetId& detid) const {

  return isSelected(detid.rawId());

}

bool DetIdSelector::operator()(const unsigned int& rawid) const {

  return isSelected(rawid);

}



