#ifndef _HISTO_
#define _HISTO_

#include <string>

class histo{
public:
	histo();
	~histo();
	void setname(std::string );
	void setbins(int );
	void setmin(int );
	void setmax(int );
	
	std::string* getname(); 
	int getbins();
	int getmin();
	int getmax();
	
protected:
	std::string *name_;
	int bins_;
	int min_;
	int max_;
	

};
#endif
