#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"
#include "CondCore/ORA/test/TestBase.h"
#include "classes.h"
//
#include <iostream>
#include <stdexcept>
#include <cstring>
// externals

#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "CoralBase/Blob.h"

using namespace testORA;

class PrimitiveContainerStreamingService : public ora::IBlobStreamingService {

  public:

    PrimitiveContainerStreamingService();
    
    virtual ~PrimitiveContainerStreamingService();

    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const edm::TypeWithDict& classDictionary, bool );

    void read( const coral::Blob& blobData, void* addressOfContainer, const edm::TypeWithDict& classDictionary );
};

PrimitiveContainerStreamingService::PrimitiveContainerStreamingService(){
}

PrimitiveContainerStreamingService::~PrimitiveContainerStreamingService(){
}

boost::shared_ptr<coral::Blob> PrimitiveContainerStreamingService::write( const void* addressOfInputData,
                                                                          const edm::TypeWithDict& type,
                                                                          bool ){
  // The actual object
  edm::ObjectWithDict theContainer( type, const_cast<void*>( addressOfInputData ) );

  // Retrieve the size of the container
  edm::FunctionWithDict sizeMethod = type.functionMemberByName( "size" );
  if ( ! sizeMethod )
    throw std::runtime_error( "No size method is defined for the container" );
  size_t containerSize = 0;
  edm::ObjectWithDict sizeObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(size_t)), &containerSize );
  sizeMethod.invoke(theContainer, &sizeObj);
  
  // Retrieve the element size
  edm::FunctionWithDict beginMethod = type.functionMemberByName( "begin" );
  if ( ! beginMethod )
    throw std::runtime_error( "No begin method is defined for the container" );
  edm::TypeWithDict iteratorType = beginMethod.finalReturnType();
  edm::FunctionWithDict dereferenceMethod = iteratorType.functionMemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw std::runtime_error( "Could not retrieve the dereference method of the container's iterator" );
  size_t elementSize = dereferenceMethod.finalReturnType().size();

  boost::shared_ptr<coral::Blob> blob( new coral::Blob( containerSize * elementSize ) );
  // allocate the blob
  void* startingAddress = blob->startingAddress();

  // Create an iterator
  edm::TypeWithDict retType2 =  beginMethod.finalReturnType();
  char* retbuf2 = ::new char[retType2.size()];
  edm::ObjectWithDict iteratorObject(retType2, retbuf2);
  beginMethod.invoke( edm::ObjectWithDict( type, const_cast< void * > ( addressOfInputData ) ), &iteratorObject );

  // Loop over the elements of the container
  edm::FunctionWithDict incrementMethod = iteratorObject.typeOf().functionMemberByName( "operator++" );
  if ( ! incrementMethod )
    throw std::runtime_error( "Could not retrieve the increment method of the container's iterator" );

  for ( size_t i = 0; i < containerSize; ++i ) {

    void* elementAddress = 0;
    edm::ObjectWithDict elemAddrObj = edm::ObjectWithDict( edm::TypeWithDict(typeid(void*)), &elementAddress );
    dereferenceMethod.invoke( iteratorObject, &elemAddrObj);
    ::memcpy( startingAddress, elementAddress, elementSize );
    char* cstartingAddress = static_cast<char*>( startingAddress );
    cstartingAddress += elementSize;
    startingAddress = cstartingAddress;

    incrementMethod.invoke( iteratorObject, 0);
  }

  // Destroy - and deallocate - the iterator
  iteratorObject.destruct(true);
  return blob;  
}

void PrimitiveContainerStreamingService::read( const coral::Blob& blobData,
                                               void* addressOfContainer,
                                               const edm::TypeWithDict& type ){
  // Retrieve the element size
  edm::FunctionWithDict beginMethod = type.functionMemberByName( "begin" );
  if ( ! beginMethod )
    throw std::runtime_error( "No begin method is defined for the container" );
  edm::TypeWithDict iteratorType = beginMethod.finalReturnType();
  edm::FunctionWithDict dereferenceMethod = iteratorType.functionMemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw std::runtime_error( "Could not retrieve the dereference method of the container's iterator" );

  size_t elementSize = dereferenceMethod.finalReturnType().size();

  if( ! elementSize ) return;

  // Retrieve the container size
  size_t contrainerSize = blobData.size() / elementSize;

  // Retrieve the end method
  edm::FunctionWithDict endMethod = type.functionMemberByName( "end" );
  if ( ! endMethod )
   throw std::runtime_error( "Could not retrieve the end method of the container" );

  // Retrieve the insert method
  edm::FunctionWithDict insertMethod = type.functionMemberByName("insert");

  // Retrieve the clear method
  edm::FunctionWithDict clearMethod = type.functionMemberByName( "clear" );
  if ( ! clearMethod )
   throw std::runtime_error( "Could not retrieve the clear method of the container" );

  // Clear the container
  edm::ObjectWithDict containerObject( type, addressOfContainer );

  clearMethod.invoke( containerObject ,0 );

  // Fill-in the elements
  const void* startingAddress = blobData.startingAddress();
  for ( size_t i = 0; i < contrainerSize; ++i ) {
    std::vector< void* > args( 2 );

    // call container.end()
    edm::TypeWithDict retType =  endMethod.finalReturnType();
    char* retbuf = ::new char[retType.size()];
    edm::ObjectWithDict ret(retType, retbuf);
    endMethod.invoke(containerObject, &ret);
    args[0] = ret.address();
    //    delete [] retbuf;
    
    // call container.insert( container.end(), data )
    args[1] = const_cast<void*>( startingAddress );
    insertMethod.invoke( containerObject, 0, args );

    // this is required! (as it was in Reflect). 
    iteratorType.destruct( args[0] );
    
    const char* cstartingAddress = static_cast<const char*>( startingAddress );
    cstartingAddress += elementSize;
    startingAddress = cstartingAddress;
    //    delete [] retbuf1;
  }  
}

namespace ora {
  class Test7: public TestBase {
  public:
    Test7(): TestBase( "testORA_7" ){
    }

    virtual ~Test7(){
    }

    void execute( const std::string& connStr ){
      ora::Database db;
      //db.configuration().setMessageVerbosity( coral::Debug );
      PrimitiveContainerStreamingService* blobServ = new PrimitiveContainerStreamingService;
      db.configuration().setBlobStreamingService( blobServ );
      db.connect( connStr );
      ora::ScopedTransaction trans( db.transaction() );
      //creating database
      trans.start( false );
      if(!db.exists()){
	db.create();
      }

      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      if( conts.find( "testORA::SiStripNoises" )!= conts.end() ) db.dropContainer( "testORA::SiStripNoises" );
      //creating container
      db.createContainer<SB>("Cont0");
      //inserting
      ora::Container contH0 = db.containerHandle( "Cont0" );
      std::vector<boost::shared_ptr<SB> > buff;
      for( unsigned int i=0;i<10;i++){
	boost::shared_ptr<SB> obj( new SB(i) );
	contH0.insert( *obj );
	buff.push_back( obj );
      }
      contH0.flush();
      buff.clear();
      //creating another container
      ora::Container cont2 = db.createContainer<SiStripNoises>();
      //inserting
      std::vector<boost::shared_ptr<SiStripNoises> > buff2;
      for( unsigned int i=0;i<10;i++){
	boost::shared_ptr<SiStripNoises> obj( new SiStripNoises(i) );
	db.insert("testORA::SiStripNoises", *obj );
	buff2.push_back( obj );
      }
      db.flush();
      buff2.clear();
      trans.commit();
      db.disconnect();
      sleep();
      // reading back...
      db.connect( connStr );
      trans.start( true );
      contH0 = db.containerHandle( "Cont0" );
      ora::ContainerIterator iter = contH0.iterator();
      while( iter.next() ){
	boost::shared_ptr<SB> obj = iter.get<SB>();
	int seed = obj->m_intData;
	SB r(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SB different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_7");
	} else{
	  std::cout << "** Read out data for class SB with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      cont2 = db.containerHandle( "testORA::SiStripNoises" );
      iter = cont2.iterator();
      while( iter.next() ){
	boost::shared_ptr<SiStripNoises> obj = iter.get<SiStripNoises>();
	unsigned int seed = obj->m_id;
	SiStripNoises r(seed);
	if( r != *obj ){
	  std::stringstream mess;
	  mess << "Data for class SiStripNoises different from expected for seed = "<<seed;
	  ora::throwException( mess.str(),"testORA_7");
	} else{
	  std::cout << "** Read out data for class SiStripNoises with seed="<<seed<<" is ok."<<std::endl;
	}
      }
      trans.commit();
      //clean up
      trans.start( false );
      db.drop();
      trans.commit();
      db.disconnect();
    }
  };
}

int main( int argc, char** argv ){
  ora::Test7 test;
  test.run();
}

