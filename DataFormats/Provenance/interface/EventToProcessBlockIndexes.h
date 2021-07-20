#ifndef DataFormats_Provenance_EventToProcessBlockIndexes_h
#define DataFormats_Provenance_EventToProcessBlockIndexes_h

/**

\author W. David Dagenhart, created 5 January, 2021

*/

namespace edm {

  class EventToProcessBlockIndexes {
  public:
    EventToProcessBlockIndexes() {}

    unsigned int index() const { return index_; }
    void setIndex(unsigned int value) { index_ = value; }

  private:
    unsigned int index_ = 0;
  };

}  // namespace edm
#endif
