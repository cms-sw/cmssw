#ifndef TestObjects_ThingWithMerge_h
#define TestObjects_ThingWithMerge_h

namespace edmtest {

  struct ThingWithMerge {
    ~ThingWithMerge() { }
    ThingWithMerge():a() { }
    bool mergeProduct(ThingWithMerge const& newThing);
    int a;
  };

}

#endif
