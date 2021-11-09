#ifndef JetResolution_h
#define JetResolution_h

// If you want to use the JER code in standalone mode, you'll need to create a new define named 'STANDALONE'. If you use gcc for compiling, you'll need to add
// -DSTANDALONE to the command line
// In standalone mode, no reference to CMSSW exists, so the only way to retrieve resolutions and scale factors are from text files.

#include <CondFormats/JetMETObjects/interface/JetResolutionObject.h>

#ifndef STANDALONE
#include "FWCore/Utilities/interface/ESGetToken.h"
namespace edm {
  class EventSetup;
}
class JetResolutionObject;
class JetResolutionRcd;
class JetResolutionScaleFactorRcd;
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
    using Token = edm::ESGetToken<JetResolutionObject, JetResolutionRcd>;
    static const JetResolution get(const edm::EventSetup&, const Token&);
#endif

    float getResolution(const JetParameters& parameters) const;

    void dump() const { m_object->dump(); }

    // Advanced usage
    const JetResolutionObject* getResolutionObject() const { return m_object.get(); }

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
    using Token = edm::ESGetToken<JetResolutionObject, JetResolutionScaleFactorRcd>;
    static const JetResolutionScaleFactor get(const edm::EventSetup&, const Token&);
#endif

    float getScaleFactor(const JetParameters& parameters,
                         Variation variation = Variation::NOMINAL,
                         std::string uncertaintySource = "") const;

    void dump() const { m_object->dump(); }

    // Advanced usage
    const JetResolutionObject* getResolutionObject() const { return m_object.get(); }

  private:
    std::shared_ptr<JetResolutionObject> m_object;
  };

};  // namespace JME

#endif
