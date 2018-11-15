#ifndef ECAL_COND_OBJECT_CONTAINER_HH
#define ECAL_COND_OBJECT_CONTAINER_HH

#include "CondFormats/Serialization/interface/Serializable.h"

#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

template < typename T >
class EcalCondObjectContainer {
        public:
                typedef T Item;
                typedef Item value_type;
                typedef EcalCondObjectContainer<T> self;
                typedef typename std::vector<Item> Items;
                typedef typename std::vector<Item>::const_iterator const_iterator; 
                typedef typename std::vector<Item>::iterator iterator;

                EcalCondObjectContainer() {};
                ~EcalCondObjectContainer() {};

                void clear() {
                  eb_.clear();
                  ee_.clear();
                }

                inline const Items & barrelItems() const { return eb_.items(); };

                inline const Items & endcapItems() const { return ee_.items(); };

                inline const Item & barrel( size_t hashedIndex ) const {
                        return eb_.item(hashedIndex);
                }
                
                inline const Item & endcap( size_t hashedIndex ) const {
                        return ee_.item(hashedIndex);
                }

                inline void insert( std::pair<uint32_t, Item> const &a ) {
                        DetId id(a.first);
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                eb_.insert(a);
                                        }
                                        break;
                                case EcalEndcap :
                                        { 
                                                ee_.insert(a);
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        return;
                        }
                }
                
                inline const_iterator find( uint32_t rawId ) const {
                        DetId id(rawId);
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                const_iterator it = eb_.find(rawId);
                                                if ( it != eb_.end() ) {
                                                        return it;
                                                } else {
                                                        return ee_.end();
                                                }
                                        }
                                        break;
                                case EcalEndcap :
                                        { 
                                                return ee_.find(rawId);
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        return ee_.end();
                        }
                        return ee_.end();
                }

                inline const_iterator begin() const {
                        return eb_.begin();
                }

                inline const_iterator end() const {
                        return ee_.end();
                }

                inline void setValue(const uint32_t id, const Item &item) {
                        (*this)[id] = item;
                }

                inline const self & getMap() const {
                        return *this;
                }

                inline size_t size() const {
                        return eb_.size() + ee_.size();
                }
                // add coherent operator++, not needed now -- FIXME

                inline Item & operator[]( uint32_t rawId ) 
                {
                        DetId id(rawId);
                        return (id.subdetId()==EcalBarrel) ? eb_[rawId] : ee_[rawId];

                }

                inline Item operator[]( uint32_t rawId ) const {
                        DetId id(rawId);
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                return eb_[rawId];
                                        }
                                        break;
                                case EcalEndcap :
                                        { 
                                                return ee_[rawId];
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        // sizeof(Item) <= sizeof(int64_t) for all Items.
                                        return Item();
                        }
                }

                void summary(float & arg_mean_x_EB,float & arg_rms_EB,int & arg_num_x_EB,
                        float & arg_mean_x_EE,float & arg_rms_EE,int & arg_num_x_EE) const {
 
                     //std::stringstream ss;
                 
                     const int kSides       = 2;
                     const int kBarlRings   = EBDetId::MAX_IETA;
                     const int kBarlWedges  = EBDetId::MAX_IPHI;
                     const int kEndcWedgesX = EEDetId::IX_MAX;
                     const int kEndcWedgesY = EEDetId::IY_MAX;

                     /// calculate mean and sigma 
                 
                     float mean_x_EB=0;
                     float mean_xx_EB=0;
                     int num_x_EB=0;
                 
                     float mean_x_EE=0;
                     float mean_xx_EE=0;
                     int num_x_EE=0;
                 
                 
                     for (int sign=0; sign<kSides; sign++) {
                 
                        int thesign = sign==1 ? 1:-1;
                 
                        for (int ieta=0; ieta<kBarlRings; ieta++) {
                          for (int iphi=0; iphi<kBarlWedges; iphi++) {
                                EBDetId id((ieta+1)*thesign, iphi+1);

                                //float x= object()[id.rawId()];
                                float x= eb_[id.rawId()];
                                num_x_EB++;
                                mean_x_EB=mean_x_EB+x;
                                mean_xx_EB=mean_xx_EB+x*x;
                          }
                        }
                 
                        for (int ix=0; ix<kEndcWedgesX; ix++) {
                              for (int iy=0; iy<kEndcWedgesY; iy++) {
                                if (! EEDetId::validDetId(ix+1,iy+1,thesign))
                                        continue;

                                EEDetId id(ix+1,iy+1,thesign);
                                //float x=object()[id.rawId()];
                                float x=ee_[id.rawId()];
                                num_x_EE++;
                                mean_x_EE=mean_x_EE+x;
                                mean_xx_EE=mean_xx_EE+x*x;
                         
                              }//iy
                        }//ix
                 
                 
                     }
                 
                     mean_x_EB=mean_x_EB/num_x_EB;
                     mean_x_EE=mean_x_EE/num_x_EE;
                     mean_xx_EB=mean_xx_EB/num_x_EB;
                     mean_xx_EE=mean_xx_EE/num_x_EE;
                     float rms_EB=(mean_xx_EB-mean_x_EB*mean_x_EB);
                     float rms_EE=(mean_xx_EE-mean_x_EE*mean_x_EE);
                 

                        arg_mean_x_EB = mean_x_EB;
                        arg_rms_EB = rms_EB;
                        arg_num_x_EB = num_x_EB;


                        arg_mean_x_EE = mean_x_EE;
                        arg_rms_EE = rms_EE;
                        arg_num_x_EE = num_x_EE;

                     //ss << "ECAL BARREL Mean: "<< mean_x_EB <<" RMS: "<<  rms_EB << " Nchan: "<< num_x_EB<< std::endl
                     //   << "ECAL Endcap Mean: "<< mean_x_EE <<" RMS: "<<  rms_EE << " Nchan: "<< num_x_EE<< std::endl ;
         
         
                     //return ss.str();
                }
                
        private:
                
                EcalContainer< EBDetId, Item > eb_;
                EcalContainer< EEDetId, Item > ee_;

        COND_SERIALIZABLE;
};

typedef EcalCondObjectContainer<float> EcalFloatCondObjectContainer;
#endif
