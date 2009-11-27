#include "CondCore/Utilities/interface/CondBasicIter.h"

CondBasicIter::~CondBasicIter(){}



CondBasicIter::CondBasicIter(
			     const std::string & NameDB,
			     const std::string & Tag,
			     const std::string & User,
			     const std::string & Pass,
			     const std::string & nameBlob) :
  rdbms(User,Pass), db(rdbms.getDB(NameDB)), iov(db.iov(Tag))
{
}

CondBasicIter::CondBasicIter(const std::string & NameDB,
	      const std::string & Tag,
	      const std::string & auth
	      ):
  rdbms(auth), db(rdbms.getDB(NameDB)), iov(db.iov(Tag))
{
}

void CondBasicIter::create(
			   const std::string & NameDB,
			   const std::string & Tag,
			   const std::string & User,
			   const std::string & Pass,
			   const std::string & nameBlob) {

  rdbms = cond::RDBMS(User,Pass);
  db = rdbms.getDB(NameDB);
  iov = db.iov(Tag);
    
}

void CondBasicIter::create(const std::string & NameDB,
			   const std::string & Tag,
			   const std::string & auth
			   ) {
  rdbms = cond::RDBMS(auth);
  db = rdbms.getDB(NameDB);
  iov = db.iov(Tag);
}


void CondBasicIter::setRange(unsigned int min,unsigned int max){
  iov.setRange(min,max);
}

void CondBasicIter::setMin(unsigned int min){
  iov.setRange(min,0);
}

void CondBasicIter::setMax(unsigned int max){
  iov.setRange(1,max);
}



unsigned int CondBasicIter::getTime()  const {return (getStartTime()+getStopTime())/2;}

unsigned int CondBasicIter::getStartTime()  const {return (*iter).since();}

unsigned int CondBasicIter::getStopTime() const {return (*iter).till();}

std::string & CondBasicIter::getToken() const  {return (*iter).token();}


bool CondBasicIter::init() {
  iter=iov.begin();
  return iter!=iov.end();

}

bool CondBasicIter::forward(){
  ++iter;
  return iter!=iov.end();
}

bool CondBasicIter::make(){
  return load(&(*iter).db().poolCache(),(*iter).token());
}

