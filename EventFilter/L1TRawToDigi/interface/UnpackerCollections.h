#ifndef UnpackerCollections_h
#define UnpackerCollections_h

namespace edm {
   class Event;
}

namespace l1t {
   class UnpackerCollections {
      public:
         UnpackerCollections(edm::Event& e) : event_(e) {};
         virtual ~UnpackerCollections() {};
      protected:
         edm::Event& event_;
   };
}

#endif
