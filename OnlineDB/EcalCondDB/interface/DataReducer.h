#ifndef DataReducer_h
#define DataReducer_h

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"
#include "OnlineDB/EcalCondDB/interface/Tm.h"

// #include <list>

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <list>
#include <cctype>

// this class is used to reduce the DCS data to a
// reasonable amount of offline DB IOVs

template <typename T>
class DataReducer {
public:
  typedef DataReducer<T> self;
  typedef typename std::pair<EcalLogicID, T> DataItem;
  typedef typename std::map<EcalLogicID, T> DataMap;
  typedef typename std::list<std::pair<Tm, DataMap> >::iterator list_iterator;
  typedef typename std::map<EcalLogicID, T>::iterator map_iterator;

  template <typename U>
  class MyData {
  public:
    typedef MyData<U> self;
    bool operator<(const MyData& rhs) {
      Tm t1 = m_iData.first;
      Tm t2 = rhs.m_iData.first;
      long long diff_time = (t1.microsTime() - t2.microsTime());
      return (diff_time < 0);
    };

    std::pair<Tm, std::pair<EcalLogicID, U> > m_iData;
  };

  typedef typename std::list<MyData<T> >::iterator iterator;

  DataReducer() { m_printout = false; };
  ~DataReducer(){};

  static const int TIMELIMIT = 60;  // the time limit in seconds to consider two events in the same IOV creation

  void setDataList(std::list<MyData<T> > _list) {
    m_list = _list;
    m_list.sort();
  };

  void getReducedDataList(std::list<std::pair<Tm, DataMap> >* my_new_list) {
    /* *************************************** 
           to get reduced data list 
       *************************************** */

    std::cout << " we are in getReducedDataList " << std::endl;
    //  std::list< std::pair< Tm, DataMap > > my_new_list ;
    iterator i;
    std::cout << " created iterator " << std::endl;

    bool firstpass = true;
    unsigned int s_old = 0;
    for (i = m_list.begin(); i != m_list.end(); i++) {
      Tm t = (*i).m_iData.first;
      DataItem d = (*i).m_iData.second;
      bool new_time_change = true;

      DataMap the_data;
      list_iterator it_good = my_new_list->end();

      if (!firstpass) {
        list_iterator it;
        int last_state = -1;
        for (it = my_new_list->begin(); it != my_new_list->end(); ++it) {
          // check on the state

          std::pair<Tm, DataMap> pair_new_list = *it;

          Tm t_l = pair_new_list.first;
          DataMap dd = pair_new_list.second;
          map_iterator ip;
          for (ip = dd.begin(); ip != dd.end(); ++ip) {
            EcalLogicID ecid = ip->first;
            T dcs_dat = ip->second;
            if (ecid.getLogicID() == d.first.getLogicID())
              last_state = dcs_dat.getStatus();
          }

          long long diff_time = (t.microsTime() - t_l.microsTime()) / 1000000;
          if (diff_time < 0)
            diff_time = -diff_time;
          if (diff_time < TIMELIMIT) {
            // data change happened at the same moment

            new_time_change = false;
            // add data to the list
            the_data = pair_new_list.second;
            it_good = it;
          }
        }

        if (last_state != d.second.getStatus()) {
          if (!new_time_change) {
            std::pair<Tm, DataMap> pair_new_list = *it_good;
            Tm t_good = pair_new_list.first;
            the_data = pair_new_list.second;
            the_data.insert(d);
            std::pair<Tm, DataMap> pair_new_good;
            pair_new_good.first = t_good;
            pair_new_good.second = the_data;

            my_new_list->erase(it_good);
            my_new_list->push_back(pair_new_good);

          } else if (new_time_change) {
            std::pair<Tm, DataMap> p_new;
            p_new.first = t;
            DataMap a_map;
            a_map.insert(d);
            p_new.second = a_map;
            my_new_list->push_back(p_new);
          }
        }
        list_iterator it3;
        if (my_new_list->size() > s_old) {
          s_old = my_new_list->size();
          if (m_printout) {
            std::cout << "************" << std::endl;
            for (it3 = my_new_list->begin(); it3 != my_new_list->end(); ++it3) {
              std::pair<Tm, DataMap> pair_new_list3 = *it3;
              Tm t3 = pair_new_list3.first;
              std::cout << " T =" << t3.str() << std::endl;
            }
            std::cout << "************" << std::endl;
          }
        }

      } else {
        // first pass write it anyway
        std::pair<Tm, DataMap> p_new;
        p_new.first = t;
        DataMap a_map;
        a_map.insert(d);
        p_new.second = a_map;
        my_new_list->insert(my_new_list->begin(), p_new);
        firstpass = false;
      }
    }

    if (m_printout) {
      list_iterator it3;
      for (it3 = my_new_list->begin(); it3 != my_new_list->end(); ++it3) {
        std::pair<Tm, DataMap> pair_new_list3 = *it3;
        Tm t3 = pair_new_list3.first;
        std::cout << " T =" << t3.str() << std::endl;
      }
    }
  };

private:
  //  std::list< std::pair< Tm, DataItem >  > m_list;
  std::list<MyData<T> > m_list;
  bool m_printout;
};

#endif
