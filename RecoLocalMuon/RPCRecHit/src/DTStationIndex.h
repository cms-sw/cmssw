#ifndef RecoLocalMuon_RPCRecHit_DTStationIndex_h
#define RecoLocalMuon_RPCRecHit_DTStationIndex_h

class DTStationIndex{
public: 
  DTStationIndex():_region(0),_wheel(0),_sector(0),_station(0) {}

  DTStationIndex(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station) {}

  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}

  bool operator<(const DTStationIndex& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }

private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};

#endif // RecoLocalMuon_RPCRecHit_DTStationIndex_h
