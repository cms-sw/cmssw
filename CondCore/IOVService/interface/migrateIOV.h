#ifndef IOVService_migrateIOV_H
#define IOVService_migrateIOV_H


namespace cond {

  class IOV;
  class IOVSequence;

  IOVSequence * migrateIOV(IOV const & iov);
  IOV * backportIOV(IOVSequence cons& sequence);

}

#endif //  IOVService_migrateIOV_H
