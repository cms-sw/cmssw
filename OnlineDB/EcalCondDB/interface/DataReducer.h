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



// this class is used to reduce the DCS data to a 
// reasonable amount of offline DB IOVs 

template < typename T > 
class DataReducer
{

 public:

  typedef DataReducer<T> self;
  typedef typename std::pair< EcalLogicID, T >  DataItem;
  typedef typename std::map<  EcalLogicID, T >  DataMap ;
  typedef typename std::list< std::pair< Tm, DataItem > >::iterator iterator;
  typedef typename  std::list< std::pair< Tm, DataMap > >::iterator list_iterator; 


  DataReducer() {};
  ~DataReducer() {};

  static const int TIMELIMIT=60; // the time limit in seconds to consider two events in the same IOV creation

  void setDataList( std::list< std::pair< Tm, DataItem > > _list ){ m_list = _list;};

  void getReducedDataList(std::list< std::pair< Tm, DataMap > >* my_new_list) {
    /* *************************************** 
           to get reduced data list 
       *************************************** */

    std::cout << " we are in getReducedDataList "<< std::endl; 
    //  std::list< std::pair< Tm, DataMap > > my_new_list ;
    iterator i;
    std::cout << " created iterator "<< std::endl; 

    bool firstpass=true; 

    for ( i=m_list.begin(); i!=m_list.end(); i++){


      Tm t = (*i).first;
      DataItem d = (*i).second;
      bool new_time_change=true;

      if(!firstpass) {
	list_iterator it; 
	for(it =my_new_list->begin(); it!= my_new_list->end(); ++it) {

	  
	  std::pair< Tm, DataMap > pair_new_list = *it;

	  
	  Tm t_l = pair_new_list.first;

	  DataMap the_data = pair_new_list.second;
	  

	  int diff_time= ((int)t.microsTime() - (int)t_l.microsTime()) /1000000  ;
	  if(diff_time < 0) diff_time= - diff_time; 
	  if(  diff_time  < TIMELIMIT ) {
	    // data change happened at the same moment
	    // TO BE DONE : add a a check that the state is not equal to the previous one 
	    
	    new_time_change=false;
	    // add data to the list

	    
	    the_data.insert(d);
	  }
	  
	}

	if(new_time_change) {
	  std::pair< Tm, DataMap >  p_new;
	  p_new.first=t;
	  DataMap a_map;
	  a_map.insert( d );
	  p_new.second=a_map;
	  my_new_list->push_back( p_new );
	}
      } else {
	// first pass write it anyway 
	  std::pair< Tm, DataMap >  p_new;
	  p_new.first=t;
	  DataMap a_map;
	  a_map.insert( d );
	  p_new.second=a_map;
	  my_new_list->push_back( p_new );
	  firstpass=false; 
	  std::cout <<"first t= "<< t.str()<<std::endl;
      }

    }
    //    std::list< std::pair< Tm, DataMap > >* result= & my_new_list;
    // return result;
  };


 private:
  std::list< std::pair< Tm, DataItem >  > m_list;

};


#endif

