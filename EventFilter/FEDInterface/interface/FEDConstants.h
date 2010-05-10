#ifndef EVF_FEDINTERFACE_FEDCONSTANTS_H
#define EVF_FEDINTERFACE_FEDCONSTANTS_H

namespace evf{
  namespace evtn{
      const unsigned int SLINK_WORD_SIZE     = 8;
      const unsigned int SLINK_HALFWORD_SIZE = 4;
      const unsigned int FED_HEADER_SIZE     = SLINK_WORD_SIZE;
      const unsigned int FED_TRAILER_SIZE    = SLINK_WORD_SIZE;
  }
}
#endif
