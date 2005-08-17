/*
 * FroNTier client API
 *
 * Author: Sergey Kosyakov
 *
 * $Header: /cvs_server/repositories/EDMProto/EDMProto/Calalign/FrontierDBReader/interface/Frontier2.h,v 1.1 2005/05/06 19:49:53 wmtan Exp $
 *
 * $Id: Frontier2.h,v 1.1 2005/05/06 19:49:53 wmtan Exp $
 *
 */
#ifndef __HEADER_H_FRONTIER_FRONTIER_CPP_H_
#define __HEADER_H_FRONTIER_FRONTIER_CPP_H_

#include <string>
#include <vector>

extern "C"
 {
#include "frontier_client/frontier.h"
 };

namespace frontier2
{

enum encoding_t {BLOB};

class DataSource;

class Request
 {
  protected:
   friend class DataSource;
   std::string obj_name;
   std::string v;
   encoding_t enc;
   std::vector<std::string> *v_key;
   std::vector<std::string> *v_val;
   int is_meta;

  public:
   Request(const std::string& name,
           const std::string& version,
           const encoding_t& encoding):
          obj_name(name),v(version),enc(encoding),v_key(NULL),v_val(NULL),is_meta(0){};
   virtual void addKey(const std::string& key,const std::string& value);
   virtual ~Request();
 };


class MetaRequest : public Request
 {
  public:
   MetaRequest(const std::string& name,
               const std::string& version,
               const encoding_t& encoding):Request(name,version,encoding) {is_meta=1;}
   virtual void addKey(const std::string& key,const std::string& value){}
   virtual ~MetaRequest(){}
 };


int init();

// Enum sucks
typedef unsigned char BLOB_TYPE;
const BLOB_TYPE BLOB_BIT_NULL=(1<<7);

const BLOB_TYPE BLOB_TYPE_NONE=BLOB_BIT_NULL;
const BLOB_TYPE BLOB_TYPE_BYTE=0;
const BLOB_TYPE BLOB_TYPE_INT4=1;
const BLOB_TYPE BLOB_TYPE_INT8=2;
const BLOB_TYPE BLOB_TYPE_FLOAT=3;
const BLOB_TYPE BLOB_TYPE_DOUBLE=4;
const BLOB_TYPE BLOB_TYPE_TIME=5;
const BLOB_TYPE BLOB_TYPE_ARRAY_BYTE=6;
const BLOB_TYPE BLOB_TYPE_EOR=7;

const char *getFieldTypeName(BLOB_TYPE t);

class DataSource;

class AnyData
 {
  private:
   friend class DataSource;
   union
    {
     long long i8;
     double d;
     struct{char *p;unsigned int s;}str;
     int i4;
     float f;
     char b;
    } v;
   int isNull;   // I do not use "bool" here because of compatibility problems [SSK]
   int type_error;
   BLOB_TYPE t;  // The data type

   int castToInt();
   long long castToLongLong();
   float castToFloat();
   double castToDouble();
   std::string* castToString();

  public:
   AnyData(): isNull(0),type_error(FRONTIER_OK),t(BLOB_TYPE_NONE){}
   
   long long getRawI8() const {return v.i8;}
   double getRawD() const {return v.d;}
   const char *getRawStrP() const {return v.str.p;}
   unsigned getRawStrS() const {return v.str.s;}
   int getRawI4() const {return v.i4;}
   float getRawF() const {return v.f;}
   char getRawB() const {return v.b;}

   inline void set(int i4){t=BLOB_TYPE_INT4;v.i4=i4;type_error=FRONTIER_OK;}
   inline void set(long long i8){t=BLOB_TYPE_INT8;v.i8=i8;type_error=FRONTIER_OK;}
   inline void set(float f){t=BLOB_TYPE_FLOAT;v.f=f;type_error=FRONTIER_OK;}
   inline void set(double d){t=BLOB_TYPE_DOUBLE;v.d=d;type_error=FRONTIER_OK;}
   inline void set(long long t,int time){t=BLOB_TYPE_TIME;v.i8=t;type_error=FRONTIER_OK;}
   inline void set(unsigned int size,char *ptr){t=BLOB_TYPE_ARRAY_BYTE;v.str.s=size;v.str.p=ptr;type_error=FRONTIER_OK;}
   inline void setEOR(){t=BLOB_TYPE_EOR;type_error=FRONTIER_OK;}
   inline BLOB_TYPE type() const{return t;}
   inline int isEOR() const{return (t==BLOB_TYPE_EOR);}

   inline int getInt(){if(isNull) return 0;if(t==BLOB_TYPE_INT4) return v.i4; return castToInt();}
   inline long long getLongLong(){if(isNull) return 0; if(t==BLOB_TYPE_INT8 || t==BLOB_TYPE_TIME) return v.i8; return castToLongLong();}
   inline float getFloat(){if(isNull) return 0.0; if(t==BLOB_TYPE_FLOAT) return v.f; return castToFloat();}
   inline double getDouble(){if(isNull) return 0.0;if(t==BLOB_TYPE_DOUBLE) return v.d; return castToDouble();}
   std::string* getString();
   void assignString(std::string *s);

   inline void clean(){if(t==BLOB_TYPE_ARRAY_BYTE && v.str.p) {delete[] v.str.p; v.str.p=NULL;}} // Thou art warned!!!

   ~AnyData(){clean();} // Thou art warned!!!
 };



class DataSource
 {
  private:
   unsigned long channel;
   std::string *uri;
   void *internal_data;
   BLOB_TYPE last_field_type;
   int first_row;

  public:
   int err_code;
   std::string err_msg;

   DataSource();

   // If reload!=0 then all requested objects will be refreshed at all caches
   // New object copy will be obtained directly from server
   void setReload(int reload);

   // Get data for Requests
   void getData(const std::vector<const Request*>& v_req);

   // Each Request generates a payload. Payload numbers started with 1.
   // So, to get data for the first Request call setCurrentLoad(1)
   void setCurrentLoad(int n);

   // Check error for this particular payload.
   // If error happes for any payload that payload and all subsequent payloads (if any) are empty
   int getCurrentLoadError() const;
   // More detailed (hopefully) error explanation.
   const char* getCurrentLoadErrorMessage() const;

   // Data fields extractors
   // These methods change DS position to the next field
   // Benchmarking had shown that inlining functions below does not improve performance
   int getAnyData(AnyData* buf,int not_eor=1);
   int getInt();
   long getLong();
   long long getLongLong();
   double getDouble();
   float getFloat();
   long long getDate();
   std::string *getString();
   std::string *getBlob();

   void assignString(std::string *s);

   // Current pyload meta info
   unsigned int getRecNum();
   unsigned int getRSBinarySize();
   unsigned int getRSBinaryPos();
   BLOB_TYPE lastFieldType(){return last_field_type;} // Original type of the last extracted field
   BLOB_TYPE nextFieldType(); // Next field type. THIS METHOD DOES NOT CHANGE DS POSITION !!!
   inline int isEOR(){return (nextFieldType()==BLOB_TYPE_EOR);}  // End Of Record. THIS METHOD DOES NOT CHANGE DS POSITION !!!
   inline int isEOF(){return (getRSBinarySize()==getRSBinaryPos());} // End Of File
   int next();

   virtual ~DataSource();
 };

}; // namespace frontier2


#endif //__HEADER_H_FRONTIER_FRONTIER_CPP_H_

