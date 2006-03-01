#ifndef CSCRPCDigi_CSCRPCDigi_h
#define CSCRPCDigi_CSCRPCDigi_h

/**\class CSCRPCDigi
 *
 * Digi for RPC data coming thru RAT-ALCT-DDU. 
 *
 * $Date: 2005/11/02 23:25:56 $
 * $Revision: 1.1 $
 *
 * \author N. Terentiev, CMU
 */

class CSCRPCDigi{

public:

  /// Enum, structures
  /// Length of packed fields
  enum packing{rpc_size   = 3,
               pad_size   = 4,
	       bxn_size   = 4,
	       tbin_size = 4  
  };
      /// The packed digi content  
  struct PackedDigiType {
    unsigned rpc  : rpc_size;
    unsigned pad  : pad_size;
    unsigned bxn  : bxn_size;
    unsigned tbin : tbin_size;
  };

      /// The packed data as seen by the persistency - should never be used
      /// directly, only by calling data().
      /// Made public to be able to generate lcgdict, SA, 27/4/05
  struct PersistentPacking {
    unsigned int w1;
  };

  /// Constructors

  explicit CSCRPCDigi (int rpc, int pad, int bxn , int tbin);  /// from the rpc#, pad#, bxn#, tbin#
  CSCRPCDigi (PackedDigiType packed_value);  /// from a packed value
  CSCRPCDigi (const CSCRPCDigi& digi);       /// copy
  CSCRPCDigi ();                             /// default

  /// Assignment operator

  CSCRPCDigi& operator=(const CSCRPCDigi& digi);

  /// Accessors
  /// all digi content in a packed format
  PackedDigiType packedData() const { return *(data()); }
  /// get RPC
  int getRpc() const ;
  /// return pad number
  int getPad() const;
  /// return tbin number
  int getTbin() const;
  /// return BXN
  int getBXN() const;
  
  /// Prints

      /// Print content of digi
  void print() const;
      /// Print the binary representation of the digi
  void dump() const;

private:
  friend class testCSCDigis;

  /// Set, access, repack

  void set(int rpc, int pad, int bxn, int tbin);  /// set data words
  void setData(PackedDigiType p);                 /// set from a PackedDigiType
  PackedDigiType* data();                         /// access to the packed data
  const PackedDigiType* data() const;             /// const access to the packed data
  PersistentPacking persistentData;

};

#endif
