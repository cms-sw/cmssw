#include "DQM/SiStripMonitorClient/interface/SiStripXMLTags.h"

                                    // XML Tag
dqm::XMLTag::~XMLTag() {
  // Clean up memory
  for( VXMLTags::const_iterator oITER = oVTags_.begin();
       oITER != oVTags_.end();
       ++oITER) {

    delete *oITER;
  }
}

std::ostream &dqm::XMLTag::printXML( std::ostream &roOut) const {
  for( VXMLTags::const_iterator oITER = oVTags_.begin();
       oITER != oVTags_.end();
       ++oITER) {

    ( *oITER)->printXML( roOut);
  }

  return roOut;
}

std::ostream &dqm::XMLTag::printXMLLite( std::ostream &roOut) const {
  for( VXMLTags::const_iterator oITER = oVTags_.begin();
       oITER != oVTags_.end();
       ++oITER) {

    ( *oITER)->printXMLLite( roOut);
  }

  return roOut;
}

std::ostream &dqm::XMLTag::printString( std::ostream &roOut) const {
  for( VXMLTags::const_iterator oITER = oVTags_.begin();
       oITER != oVTags_.end();
       ++oITER) {

    ( *oITER)->printString( roOut);
  }

  return roOut;
}


std::ostream &dqm::XMLTag::printStringLite( std::ostream &roOut) const {
  for( VXMLTags::const_iterator oITER = oVTags_.begin();
       oITER != oVTags_.end();
       ++oITER) {

    ( *oITER)->printStringLite( roOut);
  }

  return roOut;
}

std::ostream &dqm::operator <<( std::ostream &roOut,
                                const dqm::XMLTag &roXML_TAG) {

  switch( roXML_TAG.getMode()) {
    case XMLTag::XML:
      roXML_TAG.printXML( roOut);
      break;
    case XMLTag::XML_LITE:
      roXML_TAG.printXMLLite( roOut);
      break;
    case XMLTag::STRING:
      roXML_TAG.printString( roOut);
      break;
    case XMLTag::STRING_LITE:
      roXML_TAG.printStringLite( roOut);
      break;
    default:
      break;
  }

  return roOut;
}

                                    // QTest Tag
std::ostream &dqm::XMLTagQTest::printXML( std::ostream &roOut) const {
  roOut << "<qtest name='" << cName_ << "'>"
        << cMessage_ << "</qtest>"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagQTest::printString( std::ostream &roOut) const {
  roOut << cName_    << std::endl
        << cMessage_ << std::endl;

  return roOut;
}

                                    // Path Tag
std::ostream &dqm::XMLTagPath::printXML( std::ostream &roOut) const {
  roOut << "<path>" << cPath_ << "</path>"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagPath::printString( std::ostream &roOut) const {
  roOut << cPath_ << std::endl;

  return roOut;
}

                                  // Module Tag
std::ostream &dqm::XMLTagModule::printXML( std::ostream &roOut) const {
  roOut << "<module>" << std::endl;

  XMLTag::printXML( roOut);

  roOut << "</module>" << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagModule::printString( std::ostream &roOut) const {
  XMLTag::printString( roOut);

  return roOut;
}

                                  // Modules
dqm::Modules &dqm::Modules::operator++() {
  if( !*this) {
    ++nModules_;
    lock();
  }
  
  return *this;
}

                                  // Digis
std::ostream &dqm::XMLTagDigis::printXMLLite( 
  std::ostream &roOut) const {

  roOut << "<digis>" << getModules()
        << " modules out of " << getTotModules()
        << "</digis>"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagDigis::printStringLite( 
  std::ostream &roOut) const {

  roOut << "Digis: " << getModules()
        << " modules out of " << getTotModules()
        << std::endl;

  return roOut;
}

                                  // Clusters
std::ostream &dqm::XMLTagClusters::printXMLLite( 
  std::ostream &roOut) const {

  roOut << "<clusters>" << getModules()
        << " modules out of " << getTotModules()
        << "</clusters>"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagClusters::printStringLite( 
  std::ostream &roOut) const {

  roOut << "Clusters: " << getModules()
        << " modules out of " << getTotModules()
        << std::endl;

  return roOut;
}

                                  // Warnings
std::ostream &dqm::XMLTagWarnings::printXML( 
  std::ostream &roOut) const {
    
  roOut << "<warnings>" << std::endl;

  XMLTag::printXML( roOut);

  roOut << "</warnings>" << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagWarnings::printXMLLite( 
  std::ostream &roOut) const {
    
  roOut << "<warnings>" << std::endl;

  XMLTag::printXMLLite( roOut);

  roOut << "</warnings>" << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagWarnings::printString( 
  std::ostream &roOut) const {

  roOut << "--[ Warnings ]----------------------------------------------------"
        << std::endl;

  XMLTag::printString( roOut);

  roOut << "------------------------------------------------------------------"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagWarnings::printStringLite( 
  std::ostream &roOut) const {

  roOut << "--[ Warnings ]----------------------------------------------------"
        << std::endl;

  XMLTag::printStringLite( roOut);

  roOut << "------------------------------------------------------------------"
        << std::endl;

  return roOut;
}

                                  // Errors
std::ostream &dqm::XMLTagErrors::printXML( std::ostream &roOut) const {
  roOut << "<errors>" << std::endl;

  XMLTag::printXML( roOut);

  roOut << "</errors>" << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagErrors::printXMLLite( std::ostream &roOut) const {
  roOut << "<errors>" << std::endl;

  XMLTag::printXMLLite( roOut);

  roOut << "</errors>" << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagErrors::printString( 
  std::ostream &roOut) const {

  roOut << "--[ Errors ]------------------------------------------------------"
        << std::endl;

  XMLTag::printString( roOut);

  roOut << "------------------------------------------------------------------"
        << std::endl;

  return roOut;
}

std::ostream &dqm::XMLTagErrors::printStringLite( 
  std::ostream &roOut) const {

  roOut << "--[ Errors ]------------------------------------------------------"
        << std::endl;

  XMLTag::printStringLite( roOut);

  roOut << "------------------------------------------------------------------"
        << std::endl;

  return roOut;
}
