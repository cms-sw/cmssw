#ifndef ES_COND_OBJECT_CONTAINER_HH
#define ES_COND_OBJECT_CONTAINER_HH

#include "DataFormats/EcalDetId/interface/EcalContainer.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"

template < typename T >
class ESCondObjectContainer {
        public:
                typedef T Item;
                typedef Item value_type;
                typedef ESCondObjectContainer<T> self;
                typedef typename std::vector<Item> Items;
                typedef typename std::vector<Item>::const_iterator const_iterator; 
                typedef typename std::vector<Item>::iterator iterator;

                ESCondObjectContainer() {};
                ~ESCondObjectContainer() {};

                inline
                const Items & preshowerItems() const { return es_.items(); };

                inline
                const Item & preshower( size_t hashedIndex ) const {
                        return es_.item(hashedIndex);
                }
                
                inline
                void insert( std::pair<uint32_t, Item> const &a ) {
                        DetId id(a.first);
                        switch (id.subdetId()) {
                                case EcalPreshower :
                                        { 
                                                es_.insert(a);
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
			const_iterator it = es_.end();
			if(id.subdetId()== EcalPreshower) {
			  it = es_.find(rawId);
			  if ( it != es_.end() ) {
			    return it;
			  }
			} 
			return it;
                }

                inline
                const_iterator begin() const {
                        return es_.begin();
                }

                inline
                const_iterator end() const {
                        return es_.end();
                }

                inline
                void setValue(const uint32_t id, const Item &item) {
                        (*this)[id] = item;
                }

                inline
                const self & getMap() const {
                        return *this;
                }

                inline
                size_t size() const {
                        return es_.size() ;
                }
                // add coherent operator++, not needed now -- FIXME

                inline
                Item & operator[]( uint32_t rawId ) {
                        DetId id(rawId);
                        static Item dummy;
                        switch (id.subdetId()) {
                                case EcalPreshower :
                                        { 
                                                return es_[rawId];
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
                                case EcalPreshower :
                                        { 
                                                return es_[rawId];
                                        }
                                        break;
                                default:
                                        // FIXME (add throw)
                                        return dummy;
                        }
                }
                
        private:
                EcalContainer< ESDetId, Item > es_;
};

typedef ESCondObjectContainer<float> ESFloatCondObjectContainer;
#endif
