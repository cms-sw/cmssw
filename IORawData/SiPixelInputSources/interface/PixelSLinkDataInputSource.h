#ifndef IORawDataSiPixelInputSources_PixelSLinkDataInputSource_h
#define IORawDataSiPixelInputSources_PixelSLinkDataInputSource_h

// PixelSLinkDataInputSource.hh

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/ExternalInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>

class PixelSLinkDataInputSource : public edm::ExternalInputSource {

public:

  explicit PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
				     const edm::InputSourceDescription& desc);

  virtual ~PixelSLinkDataInputSource();

  bool produce(edm::Event& event);


private:

    std::string m_file_name;
    std::fstream m_file;
    int m_fedid;

};
#endif
