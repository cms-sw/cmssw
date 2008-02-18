#ifndef CondIter_CondCachedIter_h
#define CondIter_CondCachedIter_h

#include<vector>
#include <string>
#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>

#include "CondCore/Utilities/interface/CondIter.h"


template <class T>
class CondCachedIter{
    private:
        CondIter<T> * Iterator;
      
        std::vector<const T *> m_CondCachedIter;
 
        std::vector<const cond::Ref<T> *> m_TempCache;
                
        std::vector<unsigned int> m_Run;
        std::vector<unsigned int> m_RunStop;
        std::vector<unsigned int> m_RunStart;
        
                
        std::string NameDB;
        std::string File;
        std::string User;
        std::string Pass;
        std::string nameBlob;
                        
        unsigned int now;
        unsigned int howMany;
        
        unsigned int m_startTime;
        unsigned int m_stopTime;

        
    
    public:

        CondCachedIter();
        ~CondCachedIter();
 
 /**
        tell Iter to point to a database. After this call Iter can be used.
        Direct Access to database through frontier
It needs:
        \li \c NameDB -> name of the database
        \li \c File -> Tag human-readable of the content of the database
        \li \c User -> name of the User (if you don't need to authenticate don't write anything here)
        \li \c Pass -> Password to access database (if you don't need to authenticate don't write anything here)
        \li \c nameBlob -> to handle blob type of data (if it is not needed this field has to be left empty)
  */       
        
        void create(const std::string & NameDB,
                    const std::string & File,
                    const std::string & User = "",
                    const std::string & Pass = "",
                    const std::string & nameBlob = ""
                   );

         /**
        Obtain the pointer to an object T. If it is the last T the method returns a null pointer.
        next() don't delete the previous object.
          */ 
       
        T const * next();
        
        /**
        After you call this method you can access to the first element of "m_CondCachedIter" or the first element of the list of IOVs.
         */
        void rewind();
        
        /**
        Delete the last element of the "m_CondCachedIter".
         */
        void drop();
        
        /**
        Delete all the elements of the "m_CondCachedIter".
         */
        void clear();
        
            
        /**
        Set the minimum of the range of interest of the Iterator of the IOVs: thus when you call the method next() you get from min
         */  

        void setMin(int min);
        
        /**
        Set the maximum of the range of interest of the Iterator of the IOVs: thus when you call the method next() you get up to max
         */  
        void setMax(int max);
        
        /**
        Set the range of interest of the Iterator of the IOVs: thus when you call the method next() you get from min to max
         */ 
        void setRange(int min,int max); 
      
        /**
        get the mean time of the Iterval of Validity
         */  
        unsigned int getTime();
        
        /**
        get the SINCE TIME of the Interval of Validity
         */
        unsigned int getStartTime();
        
        /**
        get the TILL TIME of the Interval of Validity
         */
        unsigned int getStopTime();

};



template <class T> CondCachedIter<T>::CondCachedIter(){
    
    Iterator = 0;
    howMany = 0;
    now = 0;
    m_startTime = 0;
    m_stopTime = 0;

}


template <class T> CondCachedIter<T>::~CondCachedIter(){

    if (Iterator) {
        Iterator->free(); //now when I delete Iterator "ref" may survive
        delete Iterator;
    }
    
      
    
    while( ! m_TempCache.empty() )
    {
        if (m_TempCache.back()) {
            delete m_TempCache.back();
        }
        m_TempCache.pop_back();
    }
 
    while( ! m_CondCachedIter.empty() )
    {
        m_CondCachedIter.pop_back();
    }
    
    
    
    

    
}





template <class T> void CondCachedIter<T>::create(const std::string & NameDB_in,const std::string & File_in,const std::string & User_in,const std::string & Pass_in,const std::string & nameBlob_in){

    NameDB = NameDB_in;
    File = File_in;
    User = User_in;
    Pass = Pass_in;
    nameBlob = nameBlob_in;
    
    if (!Iterator) Iterator = new CondIter<T>;

    Iterator->create(NameDB,File,User,Pass,nameBlob);
         
}









template <class T> void CondCachedIter<T>::rewind() {
 
    now = 0; //back at the beginning
    if (!Iterator) Iterator = new CondIter<T>;
    Iterator->create(NameDB,File,User,Pass,nameBlob);  
    Iterator->setRange(m_startTime,m_stopTime);
    m_CondCachedIter.reserve(m_stopTime-m_startTime+2);
    m_TempCache.reserve(m_stopTime-m_startTime+2);

}



template <class T> T const * CondCachedIter<T>::next(){
    //if it doesn't exist yet
    if (now==howMany){
        m_CondCachedIter.push_back(Iterator->next(0));//"0" thus not destroying the object (see Iter class)
        m_TempCache.push_back(Iterator->whatRef());
        m_Run.push_back(Iterator->getTime());
        m_RunStart.push_back(Iterator->getStartTime());
        m_RunStop.push_back(Iterator->getStopTime());        
        howMany++;
    }
    else Iterator->jump();
    
    now++; //increase the position
    return m_CondCachedIter.at(now-1);
}


template <class T> void CondCachedIter<T>::clear(){

    
    while( ! m_TempCache.empty() )
    {
        if (m_TempCache.back()) delete m_TempCache.back();
        m_TempCache.pop_back();
    }
    
    while( ! m_CondCachedIter.empty() )
    {
        m_CondCachedIter.pop_back();
    }

    m_Run.clear();
    m_RunStart.clear();
    m_RunStop.clear();
    
    now = 0;
    howMany = 0;

    rewind();
       
}


template <class T> void CondCachedIter<T>::drop(){
        
    if (now) {
        if (m_TempCache.at(now-1)) delete (m_TempCache.at(now-1));
        m_CondCachedIter.pop_back(); //it destroys the last datum inserted
        m_TempCache.pop_back();
        
        m_Run.pop_back();
        m_RunStop.pop_back();
        m_RunStart.pop_back();
        now--;
        howMany--;
        
    }  

}


template <class T> void CondCachedIter<T>::setRange(int min,int max){
    
    Iterator->setRange(min,max);
    Iterator->getRange(&m_startTime,&m_stopTime);
    std::cout << "\nMIN = " << m_startTime << "\tMAX = " << m_stopTime << std::endl;
    m_CondCachedIter.reserve(m_stopTime-m_startTime+2);
    m_TempCache.reserve(m_stopTime-m_startTime+2);
    
}
 
template <class T> void CondCachedIter<T>::setMin(int min){
    Iterator->setMin(min);
    m_startTime = Iterator->getMin();
    m_CondCachedIter.reserve(m_stopTime-m_startTime+2);
    m_TempCache.reserve(m_stopTime-m_startTime+2);

}
 
template <class T> void CondCachedIter<T>::setMax(int max){
    Iterator->setMax(max);
    m_stopTime = Iterator->getMax();
    m_CondCachedIter.reserve(m_stopTime-m_startTime+2);
    m_TempCache.reserve(m_stopTime-m_startTime+2);

}
 

template <class T> unsigned int CondCachedIter<T>::getTime(){
    if (now) return m_Run.at(now-1);
    else return 0;
}


template <class T> unsigned int CondCachedIter<T>::getStartTime(){
    if (now) return m_RunStart.at(now-1);
    else return 0;
}


template <class T> unsigned int CondCachedIter<T>::getStopTime(){
    if (now) return m_RunStop.at(now-1);
    else return 0;
}















#endif


