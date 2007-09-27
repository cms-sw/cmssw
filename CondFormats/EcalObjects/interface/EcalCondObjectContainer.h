#ifndef ECAL_COND_OBJECT_CONTAINER_HH
#define ECAL_COND_OBJECT_CONTAINER_HH

#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

template < typename T >
class EcalCondObjectContainer {
        public:
                typedef T Item;
                typedef Item value_type;
                typedef EcalCondObjectContainer<T> self;
                typedef typename EcalContainer<DetId, Item>::Items Items;
                typedef typename EcalContainer<DetId, Item>::const_iterator const_iterator;
                typedef typename EcalContainer<DetId, Item>::iterator iterator;
                EcalCondObjectContainer() {};
                ~EcalCondObjectContainer() {};

                inline
                const Items & barrelItems() const { return eb_.items(); };

                inline
                const Items & endcapItems() const { return ee_.items(); };

                inline
                const Item & barrel( size_t hashedIndex ) const {
                        return eb_.item(hashedIndex);
                }
                
                inline
                const Item & endcap( size_t hashedIndex ) const {
                        return ee_.item(hashedIndex);
                }

                inline
                void insert( std::pair<uint32_t, Item> const &a ) {
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
                
                inline
                const_iterator find( uint32_t rawId ) const {
                        DetId id(rawId);
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                return eb_.find(rawId);
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
                }

                inline
                const_iterator begin() const {
                        return eb_.begin();
                }

                inline
                const_iterator end() const {
                        return ee_.end();
                }

                inline
                void setValue(const uint32_t id, const Item &item) {
                        (*this)[id] = item;
                }

                const self & getMap() const {
                        return *this;
                }

                // add coherent operator++, not needed now -- FIXME
                

                inline
                Item & operator[]( uint32_t rawId ) {
                        DetId id(rawId);
                        static Item dummy;
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
                                        return dummy;
                        }
                }
                
                inline
                Item & operator[]( ::DetId id ) {
                        static Item dummy;
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                return eb_[id];
                                        }
                                        break;
                                case EcalEndcap :
                                        { 
                                                return ee_[id];
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        return dummy;
                        }
                }
                
                inline
                Item const & operator[]( uint32_t rawId ) const {
                        DetId id(rawId);
                        static Item dummy;
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
                                        return dummy;
                        }
                }
                
                inline
                Item const & operator[]( ::DetId id ) const {
                        static Item dummy;
                        switch (id.subdetId()) {
                                case EcalBarrel :
                                        { 
                                                return eb_[id];
                                        }
                                        break;
                                case EcalEndcap :
                                        { 
                                                return ee_[id];
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        return dummy;
                        }
                }
                
        private:
                EcalContainer< EBDetId, Item > eb_;
                EcalContainer< EEDetId, Item > ee_;
};

#endif
