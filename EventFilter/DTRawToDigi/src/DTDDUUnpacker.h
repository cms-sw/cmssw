#ifndef DTDDUUnpacker_h
#define DTDDUUnpacker_h

/** \class DTDDUUnpacker
 *  The unpacker for DTs' FED.
 *
 *  $Date: 2005/11/10 18:53:57 $
 *  $Revision: 1.1.2.1 $
 * \author M. Zanetti INFN Padova
 */


class DTDDUUnpacker {

 public:
  
  /// Constructor
  DTDDUUnpacker() {}

  /// Destructor
  virtual ~DTDDUUnpacker() {}

  /// Unpacking method
  void interpretRawData(const unsigned char* index, int datasize);

 private:


};

#endif
