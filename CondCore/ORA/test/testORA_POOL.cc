#include "CondCore/ORA/interface/Database.h"
#include "CondCore/ORA/interface/Container.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include <iostream>
//#include <typeinfo>
#include "classes.h"

#include <boost/shared_ptr.hpp>
#include "Reflex/Type.h"
#include "Reflex/Object.h"

class ReflexDeleter {
  public:
    ReflexDeleter( const Reflex::Type& type ):
      m_type( type ){
    }

    ReflexDeleter( const ReflexDeleter& rhs ):
      m_type( rhs.m_type ){
    }

    void operator()( void* ptr ){
      m_type.Destruct( ptr );
    }
    
  private:
    Reflex::Type m_type;
    
};


int main(){

  boost::shared_ptr<SimpleCl> p;
  
  {
    Reflex::Type cl = Reflex::Type::ByTypeInfo( typeid(SimpleCl) );
    boost::shared_ptr<void> ptr( cl.Construct().Address(), ReflexDeleter( cl ));

    p = boost::static_pointer_cast<SimpleCl>( ptr );
  }

  std::cout << "**** Data = "<<p->m_intData<<std::endl;
  
  /**
  
    // writing...  
  ora::Database db;
  try {

    std::string connStr( "myalias" );
    //std::string connStr( "sqlite_file:mytest.db" );
  // writing...
  db.connect( connStr );
  db.transaction().start( false );
  ora::Container cn = db.containerHandle("Cont1");
  SimpleCl sim0(16);
  cn.insert( &sim0 );
  //cn.erase( 20 );
  //cn.erase( 21 );
  std::cout << "==writing on container id="<<cn.id()<<" name="<<cn.name()<<std::endl;
  cn.flush();
  db.transaction().commit();
  db.disconnect();
  // reading
  db.connect( connStr );
  db.transaction().start( true );
  bool exists = db.exists();
  if(exists){
    std::cout << "############# POOL database does exist in "<< connStr<<"."<<std::endl;
  } else {
    std::cout << "############# POOL database does not exist in "<< connStr<<", creating it..."<<std::endl;
    return -1;
  }
  std::cout <<" ############# opening db...."<<std::endl;
  std::set< std::string > conts = db.containers();
  std::cout << conts.size() <<" ############# container(s) found."<<std::endl;
  for(std::set<std::string>::const_iterator iC = conts.begin();
      iC != conts.end(); iC++ ){
    std::cout << "############# CONT=\""<<*iC<<"\""<<std::endl;
  }
  ora::Container cont = db.containerHandle("Cont1");
  std::cout << "############# CONT size="<<cont.size()<<std::endl;
  ora::ContainerIterator iter = cont.iterator();
  while( iter.next() ){
    SimpleCl* obj0 = iter.get<SimpleCl>();
    if( obj0 ){
      unsigned int seed = obj0->m_intData;
      SimpleCl ref(seed);
      if( *obj0 != ref ){
        std::cout << "### Object retrieved with seed="<<seed<<" is different from expected."<<std::endl;
      } else {
        std::cout << "### Object retrieved with seed="<<seed<<" has been read correctly."<<std::endl;
      }
    } else {
      std::cout << "### Object retrieved is null ptr."<<std::endl;
    }
  }
  db.transaction().commit();
  db.disconnect();
  } catch ( const ora::Exception& exc ){
    std::cout << "### ############# ERROR: "<<exc.what()<<std::endl;
    }
  **/
  return 0;
}

