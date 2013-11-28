#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"

GEMSuperChamber::GEMSuperChamber() 
{
}

GEMSuperChamber::GEMSuperChamber(GEMDetId ch1, GEMDetId ch2) 
{
  if ( ch1.region() ==  ch2.region() && ch1.ring() == ch2.ring() &&
       ch1.station() == ch2.station() && ch1.chamber() == ch2.chamber() &&
       ch1.roll() == ch2.roll() )
  {
    detIds_.clear();
    detIds_.push_back(ch1);
    detIds_.push_back(ch2);
  }
}

GEMSuperChamber::~GEMSuperChamber() 
{
}

const std::vector<GEMDetId>&
GEMSuperChamber::ids() const
{
  return detIds_;
}

bool 
GEMSuperChamber::operator==(const GEMSuperChamber& sch) const 
{
  return ( this->ids().at(0) == sch.ids().at(0) &&
	   this->ids().at(1) == sch.ids().at(1) );
}

void 
GEMSuperChamber::add(GEMChamber* rl) 
{
  //const GEMDetId chId(rl->id());
  if (std::find(chambers_.begin(), chambers_.end(), rl) == chambers_.end()){
    //  detIds_.push_back(chId);
    chambers_.push_back(rl);
  }
}

const std::vector<const GEMChamber*>& 
GEMSuperChamber::chambers() const 
{
  return chambers_;
}

int
GEMSuperChamber::nChambers() const
{
  return chambers_.size();
}

const GEMChamber* 
GEMSuperChamber::chamber(GEMDetId id) const
{
 // not in this super chamber! 
 if (id.chamber()!=detIds_.at(0).chamber() &&
     id.chamber()!=detIds_.at(1).chamber()) return 0;
 
  return chamber(id.layer());
}

const GEMChamber* 
GEMSuperChamber::chamber(int isl) const 
{
  for (auto detId : detIds_){
    if (detId.layer()==isl) 
      return dynamic_cast<const GEMChamber*>(chambers_.at(isl-1));
  }
  return 0;
}
