#ifndef CondIter_CondIter_h
#define CondIter_CondIter_h


#include "CondCore/Utilities/interface/CondBasicIter.h"



#include "CondCore/DBCommon/interface/TypedRef.h"

#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"

#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
#include<vector>
#include <string>


template <class T>
class CondIter{
    private:
        

        const T* Reference;
        cond::TypedRef<T> *ref;  
        CondBasicIter bIter;
                  
    public:
   
        bool NumberOfTimes;
        
        
        CondIter();
        ~CondIter();

        
        template <class A> friend class CondCachedIter;
        
        
        
 /**
        tell Iter to point to a database. After this call CondIter can be used.
       
It needs:
        \li \c  NameDB -> name of the database
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
        As default the next() delete the previous object. Set "test = 0" in order to store the datum.
 */ 


        T const * next(int test = 1);

/**
        Set the range of interest of the Iterator of the IOVs: thus when you call the method next() you get from min to max
 */ 

        void setRange(unsigned int min,unsigned int max);
        void setRange(int min,int max); 

/**
        Set the minimum of the range of interest of the Iterator of the IOVs: thus when you call the method next() you get from min
 */  
        void setMin(unsigned int min);
        void setMin(int min);
 
/**
        Set the maximum of the range of interest of the Iterator of the IOVs: thus when you call the method next() you get up to max
 */  
 
        void setMax(unsigned int max);
        void setMax(int max);
 
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

   
/**
        It passes to the following IOV without returning anything
 */
  
        void jump();

/**
        free the object T *Reference from the class CondIter: object T *Reference isn't deleted when the class CondIter is deleted.
 */        
        void free();
  
/**
        I need this method in order to manage the memory leak. If I use method "next(0)" I mantain the Ref which may cause a memory leak if I don't delete it. In order to delete it I need its pointer: the method "whatRef" gives me the pointer.  
 */
        
        cond::TypedRef<T> const * whatRef();
        
/**
        It returns the minimum of the range of interest of the Iterator of the IOVs
 */
  
        unsigned int getMin();
  
/**
        It returns the maximum of the range of interest of the Iterator of the IOVs
 */
        unsigned int getMax();
  

/**
        It enables to retrieve the minimum and the maximum of the range of interest of the Iterator of the IOVs
 */  
        void getRange(unsigned int*,unsigned int*);

  
};


          template <class T> CondIter<T>::CondIter(){

              Reference = 0;
              ref = 0;  
          }


          template <class T> CondIter<T>::~CondIter(){

              if (ref) delete ref; 
          }


          template <class T> void CondIter<T>::create(const std::string & NameDB,const std::string & File,const std::string & User,const std::string & Pass,const std::string & nameBlob){

              bIter.create(NameDB,File,User,Pass,nameBlob);
              NumberOfTimes = true;
              ref = 0; 
              Reference = 0;

          }





 
          template <class T> T const * CondIter<T>::next(int test) {
             
              if ((bIter.iter_Min)==0 && (bIter.iter_Max)==0){
// method next without range

                  if( (bIter.ioviterator)->next() ){
       
                      std::cout<<"PayloadContainerName "<<bIter.payloadContainer<<"\n";
                      std::cout<<"since \t till \t payloadToken"<<std::endl;
                      std::cout<< (bIter.ioviterator)->validity().first<<" \t "<< (bIter.ioviterator)->validity().second<<" \t "<<(bIter.ioviterator)->payloadToken()<<std::endl; 
                      if (test && ref) {
//                 delete Reference;
                          delete ref;
                      }   
                      ref = new cond::TypedRef<T> (*(bIter.pooldb),(bIter.ioviterator)->payloadToken());
                      Reference = (const T*) ref->ptr();
                      bIter.setStartTime((bIter.ioviterator)->validity().first);
                      bIter.setStopTime((bIter.ioviterator)->validity().second);
    
                      long long int temp = (long long int) (( bIter.getStartTime()+ bIter.getStopTime()) / 2.);
                      if (temp<0) temp = - temp;
                      bIter.setTime((unsigned int) temp);
                      return Reference;
                  }
                  else {
                      std::cout << "No more data ! " << std::endl;
                      Reference = 0;
                      ref = 0;
                      return Reference;
    //return a pointer to NULL
                  }
              }
              else{    
//test to see if it is the first time
                  if (NumberOfTimes){
                      unsigned int minTemp = bIter.getMin(); 
                      for (unsigned int i=0; i< minTemp; i++){//iter_Min <= run since at least I have one IOV for each run  
    //control if the min is too high
                          if (!((bIter.ioviterator)->next())) {
                              std::cout << "No data. Minimum too high." << std::endl;
                              Reference = 0;
                              ref = 0;
                              return Reference;
    //return a pointer to NULL
                          }
                          if (((bIter.ioviterator)->validity().second)>=minTemp) i = minTemp; //Minimum reached 
                      }
                      NumberOfTimes = false;
                  }
                  if( (bIter.ioviterator)->next() ){
                      if (((bIter.ioviterator)->validity().first)>=(bIter.getMax())){
                          std::cout << "No more data in the range" << std::endl;
                          Reference = 0;
                          ref = 0;
                          return Reference;
        //return a pointer to NULL
                      }
                      else {
                          std::cout<<"PayloadContainerName "<<bIter.payloadContainer<<"\n";
                          std::cout<<"since \t till \t payloadToken"<<std::endl;
                          std::cout<< (bIter.ioviterator)->validity().first<<" \t "<< (bIter.ioviterator)->validity().second<<" \t "<<(bIter.ioviterator)->payloadToken()<<std::endl; 

    
                          if (test && ref) {
//                     delete Reference;
                              delete ref;
                          }   //test to choose if mantain the object or not
                          ref = new cond::TypedRef<T> (*(bIter.pooldb),(bIter.ioviterator)->payloadToken());
                          Reference = (const T*) ref->ptr();
                          bIter.setStartTime((bIter.ioviterator)->validity().first);
                          bIter.setStopTime((bIter.ioviterator)->validity().second);
    
                          long long int temp = (long long int) (( bIter.getStartTime()+ bIter.getStopTime()) / 2.);
                          if (temp<0) temp = - temp;
                          bIter.setTime((unsigned int) temp);   
                          return Reference;
                      }
                  }
      
                  else {
                      std::cout << "No more data ! " << std::endl;
                      Reference = 0;
                      ref = 0;
                      return Reference;
    //return a pointer to NULL
                  }
              }

          }






          template <class T> void CondIter<T>::jump() {
    
              if (NumberOfTimes){
                  unsigned int minTemp = bIter.getMin(); 
          
                  for (unsigned int i=0; i< minTemp; i++){//iter_Min <= run since at least I have one IOV for each run 
    //control if the min is too high
                      if (!((bIter.ioviterator)->next())) {
                          std::cout << "No data. Minimum too high." << std::endl;
    //return a pointer to NULL
                      }
                      if (((bIter.ioviterator)->validity().second)>=minTemp) i = minTemp; //Minimum reached 
                  }
                  NumberOfTimes = false;
              }
              else (bIter.ioviterator)->next();
          }


          template <class T> void CondIter<T>::free(){
              Reference = 0;
              ref = 0;
          }




          template <class T> cond::TypedRef<T> const * CondIter<T>::whatRef() {

              return ref;
    
          }








          template <class T> void CondIter<T>::setRange(unsigned int min,unsigned int max){
              bIter.setRange(min,max);
          }
 
          template <class T> void CondIter<T>::setMin(unsigned int min){
              bIter.setMin(min);
          }
 
          template <class T> void CondIter<T>::setMax(unsigned int max){
              bIter.setMax(max);
          }

          template <class T> void CondIter<T>::setRange(int min,int max){
              bIter.setRange(min,max);
          }
 
          template <class T> void CondIter<T>::setMin(int min){
              bIter.setMin(min);
          }
 
          template <class T> void CondIter<T>::setMax(int max){
              bIter.setMax(max);
          }
 

          template <class T> unsigned int CondIter<T>::getMin(){return bIter.getMin();}
  
          template <class T> unsigned int CondIter<T>::getMax(){return bIter.getMax();}
  
          template <class T> void CondIter<T>::getRange(unsigned int * Min_out,unsigned int * Max_out){
              *Min_out = bIter.getMin();
              *Max_out = bIter.getMax();
          }


          template <class T> unsigned int CondIter<T>::getTime(){return bIter.getTime();}
  
          template <class T> unsigned int CondIter<T>::getStartTime(){return bIter.getStartTime();}
  
          template <class T> unsigned int CondIter<T>::getStopTime(){return bIter.getStopTime();}
  
#endif

