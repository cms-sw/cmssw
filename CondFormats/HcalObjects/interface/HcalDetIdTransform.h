#ifndef CondFormats_HcalObjects_HcalDetIdTransform_h
#define CondFormats_HcalObjects_HcalDetIdTransform_h

class HcalDetId;

namespace HcalDetIdTransform
{
    // When you add more transforms, add codes for them at the end
    // of the enum, just before "N_TRANSFORMS". Don't forget to adjust
    // the "transform" function accordingly.
    enum {
        RAWID = 0,    // Raw detector id
        IETA,         // ieta() + shift
        IETAABS,      // ietaAbs()
        SUBDET,       // subdetId()
        IETADEPTH,    // maps ieta() and depth() into a unique number
        IETAABSDEPTH, // maps ietaAbs() and depth() into a unique number
        N_TRANSFORMS
    };

    // Transform the detector id
    unsigned transform(const HcalDetId& id, unsigned transformCode);

    // The following function will throw an exception
    // if the transform code is invalid
    void validateCode(unsigned transformCode);
}

#endif // CondFormats_HcalObjects_HcalDetIdTransform_h
