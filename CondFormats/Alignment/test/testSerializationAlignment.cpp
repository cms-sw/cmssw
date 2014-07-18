#include "CondFormats/Serialization/interface/Test.h"

#include "../src/headers.h"

int main()
{
    testSerialization<AlignTransform>();
    testSerialization<AlignTransformError>();
    testSerialization<AlignmentErrors>();
    testSerialization<AlignmentSurfaceDeformations>();
    testSerialization<AlignmentSurfaceDeformations::Item>();
    testSerialization<Alignments>();
    testSerialization<SurveyError>();
    testSerialization<SurveyErrors>();
    testSerialization<std::vector<AlignTransform>>();
    testSerialization<std::vector<AlignTransformError>>();
    testSerialization<std::vector<AlignmentSurfaceDeformations::Item>>();
    testSerialization<std::vector<SurveyError>>();
    testSerialization<uint32_t>();

    return 0;
}
