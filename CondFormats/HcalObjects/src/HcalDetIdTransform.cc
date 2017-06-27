#include "CondFormats/HcalObjects/interface/HcalDetIdTransform.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace HcalDetIdTransform
{
    unsigned transform(const HcalDetId& id, const unsigned transformCode)
    {
        static const int ietaShift = 1024;
        static const int maxHcalDepth = 64;

        if (transformCode >= N_TRANSFORMS)
            throw cms::Exception("In HcalDetIdTransform::transform:"
                                 " invalid transform code");
        unsigned t = 0;
        switch (transformCode)
        {
        case RAWID:
            t = id.rawId();
            break;

        case IETA:
            t = id.ieta() + ietaShift;
            break;

        case IETAABS:
            t = id.ietaAbs();
            break;

        case SUBDET:
            t = id.subdetId();
            break;

        case IETADEPTH:
            t = (id.ieta() + ietaShift)*maxHcalDepth + id.depth();
            break;

        case IETAABSDEPTH:
            t = id.ietaAbs()*maxHcalDepth + id.depth();
            break;

        default:
            throw cms::Exception("In HcalDetIdTransform::transform:"
                                 " unhandled switch clause. This is a bug."
                                 " Please report.");
        }
        return t;
    }

    void validateCode(const unsigned transformCode)
    {
        if (transformCode >= N_TRANSFORMS)
            throw cms::Exception("In HcalDetIdTransform::validateCode:"
                                 " invalid transform code");
    }
}
