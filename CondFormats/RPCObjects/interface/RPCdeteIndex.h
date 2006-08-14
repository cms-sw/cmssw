#ifndef RPCObject_RPCdeteIndex_h
#define RPCObject_RPCdeteIndex_h
/** \class RPCdeteIndex 
 *
 *  Description"
 *       Class to index the geographical position of the strip
 *  \author Marcello Maggi -- INFN Bari
 *
 */

class RPCdeteIndex{
 public:
  RPCdeteIndex();
  RPCdeteIndex(int regionId, int diskId, int stationId, int sectorId, 
	       int layerId, int subsectorId, int rollId, int stripId);
  bool operator <(const RPCdeteIndex& dinx) const;

  int region() const;
  int disk() const;
  int station() const;
  int sector() const;
  int layer() const; 
  int subsector() const;
  int roll() const;
  int strip() const;

 private:
  int reg;
  int dis;
  int sta;
  int sec;
  int lay;
  int sub;
  int rol;
  int str;
};
#endif
