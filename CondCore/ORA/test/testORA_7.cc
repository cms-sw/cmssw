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
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "CoralBase/Blob.h"

using namespace testORA;

class PrimitiveContainerStreamingService : public ora::IBlobStreamingService {

  public:

    PrimitiveContainerStreamingService();
    
    virtual ~PrimitiveContainerStreamingService();

    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const Reflex::Type& classDictionary, bool );

    void read( const coral::Blob& blobData, void* addressOfContainer, const Reflex::Type& classDictionary );
};

PrimitiveContainerStreamingService::PrimitiveContainerStreamingService(){
}

PrimitiveContainerStreamingService::~PrimitiveContainerStreamingService(){
}

boost::shared_ptr<coral::Blob> PrimitiveContainerStreamingService::write( const void* addressOfInputData,
                                                                          const Reflex::Type& type,
                                                                          bool ){
  // The actual object
  Reflex::Object theContainer( type, const_cast<void*>( addressOfInputData ) );

  // Retrieve the size of the container
  Reflex::Member sizeMethod = type.MemberByName( "size" );
  if ( ! sizeMethod )
    throw std::runtime_error( "No size method is defined for the container" );
  size_t containerSize = 0;
  sizeMethod.Invoke(theContainer, containerSize);
  
  // Retrieve the element size
  Reflex::Member beginMethod = type.MemberByName( "begin" );
  if ( ! beginMethod )
    throw std::runtime_error( "No begin method is defined for the container" );
  Reflex::Type iteratorType = beginMethod.TypeOf().ReturnType();
  Reflex::Member dereferenceMethod = iteratorType.MemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw std::runtime_error( "Could not retrieve the dereference method of the container's iterator" );
  size_t elementSize = dereferenceMethod.TypeOf().ReturnType().SizeOf();

  boost::shared_ptr<coral::Blob> blob( new coral::Blob( containerSize * elementSize ) );
  // allocate the blob
  void* startingAddress = blob->startingAddress();

  // Create an iterator
  Reflex::Type retType2 =  beginMethod.TypeOf().ReturnType();
  char* retbuf2 = ::new char[retType2.SizeOf()];
  Reflex::Object iteratorObject(retType2, retbuf2);
  beginMethod.Invoke( Reflex::Object( type, const_cast< void * > ( addressOfInputData ) ), &iteratorObject );

  // Loop over the elements of the container
  Reflex::Member incrementMethod = iteratorObject.TypeOf().MemberByName( "operator++" );
  if ( ! incrementMethod )
    throw std::runtime_error( "Could not retrieve the increment method of the container's iterator" );

  for ( size_t i = 0; i < containerSize; ++i ) {

    void* elementAddress = 0;
    dereferenceMethod.Invoke( iteratorObject, elementAddress);
    ::memcpy( startingAddress, elementAddress, elementSize );
    char* cstartingAddress = static_cast<char*>( startingAddress );
    cstartingAddress += elementSize;
    startingAddress = cstartingAddress;

    incrementMethod.Invoke( iteratorObject, 0);
  }

  // Destroy the iterator
  iteratorObject.Destruct();
  return blob;  
}

void PrimitiveContainerStreamingService::read( const coral::Blob& blobData,
                                               void* addressOfContainer,
                                               const Reflex::Type& type ){
  // Retrieve the element size
  Reflex::Member beginMethod = type.MemberByName( "begin" );
  if ( ! beginMethod )
    throw std::runtime_error( "No begin method is defined for the container" );
  Reflex::Type iteratorType = beginMethod.TypeOf().ReturnType();
  Reflex::Member dereferenceMethod = iteratorType.MemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw std::runtime_error( "Could not retrieve the dereference method of the container's iterator" );

  size_t elementSize = dereferenceMethod.TypeOf().ReturnType().SizeOf();

  // Retrieve the container size
  size_t contrainerSize = blobData.size() / elementSize;

  // Retrieve the end method
  Reflex::Member endMethod = type.MemberByName( "end" );
  if ( ! endMethod )
   throw std::runtime_error( "Could not retrieve the end method of the container" );

  // Retrieve the insert method
  Reflex::Member insertMethod;
  for( unsigned int i = 0; i < type.FunctionMemberSize();i++){
    Reflex::Member im = type.FunctionMemberAt(i);
    if( im.Name() != std::string( "insert" ) ) continue;
    if( im.TypeOf().FunctionParameterSize() != 2) continue;
    insertMethod = im;
    break;
  }

  // Retrieve the clear method
  Reflex::Member clearMethod = type.MemberByName( "clear" );
  if ( ! clearMethod )
   throw std::runtime_error( "Could not retrieve the clear method of the container" );

  // Clear the container
  Reflex::Object containerObject( type, addressOfContainer );

  clearMethod.Invoke( containerObject ,0 );

  // Fill-in the elements
  const void* startingAddress = blobData.startingAddress();
  for ( size_t i = 0; i < contrainerSize; ++i ) {
    std::vector< void* > args( 2 );

    // call container.end()
    Reflex::Type retType =  endMethod.TypeOf().ReturnType();
    char* retbuf = ::new char[retType.SizeOf()];
    Reflex::Object ret(retType, retbuf);
    endMethod.Invoke(containerObject, &ret);
    args[0] = ret.Address();
    //    delete [] retbuf;
    
    // call container.insert( container.end(), data )
    args[1] = const_cast<void*>( startingAddress );
    insertMethod.Invoke( containerObject, 0, args );

    // this is required! (as it was in Reflect). 
    iteratorType.Destruct( args[0] );
    
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
      trans.start( false );
      if(!db.exists()){
	db.create();
      }

      std::set< std::string > conts = db.containers();
      if( conts.find( "Cont0" )!= conts.end() ) db.dropContainer( "Cont0" );
      if( conts.find( "testORA::SiStripNoises" )!= conts.end() ) db.dropContainer( "testORA::SiStripNoises" );
      //
      db.createContainer<SB>("Cont0");

      ora::Container contH0 = db.containerHandle( "Cont0" );
      std::vector<boost::shared_ptr<SB> > buff;
      for( unsigned int i=0;i<10;i++){
	boost::shared_ptr<SB> obj( new SB(i) );
	contH0.insert( *obj );
	buff.push_back( obj );
      }
      contH0.flush();
      buff.clear();

      ora::Container cont2 = db.createContainer<SiStripNoises>();

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
      ::sleep(1);
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

