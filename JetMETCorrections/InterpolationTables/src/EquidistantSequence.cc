#include <cmath>
#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/EquidistantSequence.h"

namespace npstat {
    EquidistantInLinearSpace::EquidistantInLinearSpace(
        const double minScale, const double maxScale, const unsigned nScales)
        : std::vector<double>()
    {
        switch (nScales)
        {
        case 0:
        break;

        case 1:
        {
            this->reserve(nScales);
            const double sc = (minScale == maxScale ? minScale :
                               (minScale + maxScale)/2.0);
            push_back(sc);
        }
        break;

        default:
        {
            this->reserve(nScales);
            const double step = (maxScale - minScale)/(nScales - 1);
            push_back(minScale);
            for (unsigned i=1; i<nScales - 1; ++i)
                push_back(minScale + i*step);
            push_back(maxScale);
        }
        break;
        }
    }

    EquidistantInLogSpace::EquidistantInLogSpace(
        const double minScale, const double maxScale, const unsigned nScales)
        : std::vector<double>()
    {
        if (nScales)
            if (!(minScale > 0.0 && maxScale > 0.0))
                throw npstat::NpstatInvalidArgument(
                    "In npstat::EquidistantInLogSpace constructor: "
                    "minimum and maximum scales must be positive");
        switch (nScales)
        {
        case 0:
        break;

        case 1:
        {
            this->reserve(nScales);
            const double sc = (minScale == maxScale ? minScale :
                               sqrt(minScale*maxScale));
            push_back(sc);
        }
        break;

        default:
        {
            this->reserve(nScales);
            const double logmax = log(maxScale);
            const double logmin = log(minScale);
            const double logstep = (logmax - logmin)/(nScales - 1);
            push_back(minScale);
            for (unsigned i=1; i<nScales - 1; ++i)
                push_back(exp(logmin + i*logstep));
            push_back(maxScale);
        }
        break;
        }
    }
}
