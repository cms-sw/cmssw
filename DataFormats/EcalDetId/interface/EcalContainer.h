#ifndef ECALDETID_ECALCONTAINER_H
#define ECALDETID_ECALCONTAINER_H

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include <utility>
#include <algorithm>

// #include <iostream>



/* a generic container for ecal items
 * provides access by hashedIndex and by DetId...
 */

template<typename DetId, typename T>
class EcalContainer {

        public:

                typedef EcalContainer<DetId, T> self;
                typedef T Item;
                typedef Item value_type;
                typedef typename std::vector<Item> Items; 
                typedef typename std::vector<Item>::const_iterator const_iterator;
                typedef typename std::vector<Item>::iterator iterator;

                   
                EcalContainer() {checkAndResize();}

                void insert(std::pair<uint32_t, Item> const &a) {
                        (*this)[a.first] = a.second;
                }

                inline const Item & item(size_t hashid) const {
                        return m_items[hashid];
                }

                inline const Items & items() const {
                        return m_items;
                }

                inline Item & operator[](uint32_t rawId) {
		  checkAndResize();
		  static Item dummy;
		  DetId id(rawId);
		  if ( !isValidId(id) ) return dummy;
		  return m_items[id.hashedIndex()];
                }


		void checkAndResize() {
		  if (m_items.size()==0) {
		    //		    std::cout << "resizing to " << DetId::kSizeForDenseIndexing << std::endl;
		    m_items.resize(DetId::kSizeForDenseIndexing);
		  }
		}


		void checkAndResize( size_t priv_size ) {
		  // this method allows to resize the vector to a specific size forcing a specific value
		  if (m_items.size()==0) {
		    //		    std::cout << "resizing to " << priv_size << std::endl;
		    m_items.resize(priv_size);
		  }
		}

                inline Item const & operator[](uint32_t rawId) const {
		  //                        if (m_items.size()==0) {
		  //	  std::cout << "resizing to " << DetId::kSizeForDenseIndexing << std::endl;
                  //              m_items.resize((size_t) DetId::kSizeForDenseIndexing);
                  //      }
                        static Item dummy;
                        DetId id(rawId);
                        if ( !isValidId(id) ) return dummy;
                        return m_items[id.hashedIndex()];
                }

                inline const_iterator find(uint32_t rawId) const {
                        DetId ib(rawId);
                        if ( !isValidId(ib) ) return m_items.end();
                        return m_items.begin() + ib.hashedIndex();
                }

                inline const_iterator begin() const {
                        return m_items.begin();
                }

                inline const_iterator end() const {
                        return m_items.end();
                }

                inline size_t size() const {
                        return m_items.size();
                }

        private:

                // not protected on EB <--> EE swap -- FIXME?
                inline bool isValidId(const DetId id) const {
                        return id.det() == ::DetId::Ecal;
                };

                std::vector<Item> m_items;

};



#endif // ECALCONTAINER
