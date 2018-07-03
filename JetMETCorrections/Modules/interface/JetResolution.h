#ifndef JetResolution_h
#define JetResolution_h

// If you want to use the JER code in standalone mode, you'll need to create a new define named 'STANDALONE'. If you use gcc for compiling, you'll need to add
// -DSTANDALONE to the command line
// In standalone mode, no reference to CMSSW exists, so the only way to retrieve resolutions and scale factors are from text files.

#include <CondFormats/JetMETObjects/interface/JetResolutionObject.h>

#ifndef STANDALONE
namespace edm {
    class EventSetup;
}
#endif


namespace JME {
    class JetResolution {
        public:
            JetResolution(const std::string& filename);
            JetResolution(const JetResolutionObject& object);
            JetResolution() {
                // Empty
            }

#ifndef STANDALONE
            static const JetResolution get(const edm::EventSetup&, const std::string&);
#endif

            float getResolution(const JetParameters& parameters) const;

            void dump() const {
                m_object->dump();
            }

            // Advanced usage
            const JetResolutionObject* getResolutionObject() const {
                return m_object.get();
            }

        private:
            std::shared_ptr<JetResolutionObject> m_object;
    };

    class JetResolutionScaleFactor {
        public:
            JetResolutionScaleFactor(const std::string& filename);
            JetResolutionScaleFactor(const JetResolutionObject& object);
            JetResolutionScaleFactor() {
                // Empty
            }

#ifndef STANDALONE
            static const JetResolutionScaleFactor get(const edm::EventSetup&, const std::string&);
#endif

            float getScaleFactor(const JetParameters& parameters, Variation variation = Variation::NOMINAL) const;

            void dump() const {
                m_object->dump();
            }

            // Advanced usage
            const JetResolutionObject* getResolutionObject() const {
                return m_object.get();
            }

        private:
            std::shared_ptr<JetResolutionObject> m_object;
    };

};

#endif

