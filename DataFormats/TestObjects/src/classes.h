#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/AssociationMapHelpers.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/TestObjects/interface/SchemaEvolutionTestObjects.h"
#include "DataFormats/TestObjects/interface/MissingDictionaryTestObject.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"
#include "DataFormats/TestObjects/interface/ThingWithPostInsert.h"
#include "DataFormats/TestObjects/interface/ThingWithDoNotSort.h"
#include "DataFormats/TestObjects/interface/TrackOfThings.h"
#include "DataFormats/TestObjects/interface/TrackOfDSVThings.h"

#include "DataFormats/TestObjects/interface/StreamTestSimple.h"
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"

#include "DataFormats/TestObjects/interface/TableTest.h"

#include "DataFormats/TestObjects/interface/DeleteEarly.h"

#include "DataFormats/TestObjects/interface/VectorVectorTop.h"

#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RandomNumberGeneratorState.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <list>
#include <algorithm>

// related to SchemaEvolutionTestObjects.h
#ifndef DataFormats_TestObjects_USE_OLD

// The following is from an example by Jakob Blomer from the ROOT team
namespace edmtest::compat {
  template <typename T>
  struct deprecated_auto_ptr {
    // We use compat_auto_ptr only to assign the wrapped raw pointer
    // to a unique pointer in an I/O customization rule.
    // Therefore, we don't delete on destruction (because ownership
    // gets transferred to the unique pointer).

    // ~deprecated_auto_ptr() { delete _M_ptr; }

    T *_M_ptr = nullptr;
  };
}  // namespace edmtest::compat

#endif
