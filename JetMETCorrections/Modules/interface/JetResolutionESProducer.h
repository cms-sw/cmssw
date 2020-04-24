#ifndef JERESProducer_h
#define JERESProducer_h

//
// Author: Sébastien Brochet
//

#include <string>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/JetResolutionRcd.h"
#include "CondFormats/DataRecord/interface/JetResolutionScaleFactorRcd.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"

class JetResolutionESProducer : public edm::ESProducer
{
    private:
        std::string m_label;

    public:
        JetResolutionESProducer(edm::ParameterSet const& fConfig) 
        {
            m_label = fConfig.getParameter<std::string>("label");
            setWhatProduced(this, m_label);
        }

        ~JetResolutionESProducer() {}

        std::shared_ptr<JME::JetResolution> produce(JetResolutionRcd const& iRecord) {
            
            // Get object from record
            edm::ESHandle<JME::JetResolutionObject> jerObjectHandle;
            iRecord.get(m_label, jerObjectHandle);

            // Convert this object to a JetResolution object
            JME::JetResolutionObject const& jerObject = (*jerObjectHandle);
            return std::make_shared<JME::JetResolution>(jerObject);
        }
};

class JetResolutionScaleFactorESProducer : public edm::ESProducer
{
    private:
        std::string m_label;

    public:
        JetResolutionScaleFactorESProducer(edm::ParameterSet const& fConfig)
        {
            m_label = fConfig.getParameter<std::string>("label");
            setWhatProduced(this, m_label);
        }

        ~JetResolutionScaleFactorESProducer() {}

        std::shared_ptr<JME::JetResolutionScaleFactor> produce(JetResolutionScaleFactorRcd const& iRecord) {
            
            // Get object from record
            edm::ESHandle<JME::JetResolutionObject> jerObjectHandle;
            iRecord.get(m_label, jerObjectHandle);

            // Convert this object to a JetResolution object
            JME::JetResolutionObject const& jerObject = (*jerObjectHandle);
            return std::make_shared<JME::JetResolutionScaleFactor>(jerObject);
        }
};
#endif
