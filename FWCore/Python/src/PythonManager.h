//
// Smart reference + singleton to handle Python interpreter
//
// Original Author:  Chris D Jones & Benedikt Hegner
//         Created:  Sun Jul 22 11:03:53 EST 2006
//
// $Id$

#ifndef Python_Manager_h
#define Python_Manager_h

namespace {
   class PythonManagerHandle;
   class PythonManager {
      public:
	 friend class PythonManagerHandle;
	 static PythonManagerHandle handle();

      private:
	 PythonManager() : refCount_(0) {
	    //deactivate use of signal handling
	    Py_InitializeEx(0);
	 }
	 ~PythonManager() {
	    Py_Finalize();
	 }
	 void increment() {
	    ++refCount_;
	 }

	 void decrement() {
	    --refCount_;
	    if(0==refCount_) {
	       delete this;
	    }
	 }
	 unsigned long refCount_;
   };
   
   
   class PythonManagerHandle {
      public:
	 ~PythonManagerHandle() { manager_.decrement(); }

	 PythonManagerHandle(PythonManager& iM):
	    manager_(iM) {
	    manager_.increment();
	 }

	 PythonManagerHandle(const PythonManagerHandle& iRHS) :
	    manager_(iRHS.manager_) {
	    manager_.increment();
	 }
      private:
	 const PythonManagerHandle& operator=(const PythonManagerHandle&);

	 PythonManager& manager_;
   };

   PythonManagerHandle PythonManager::handle() {
      static PythonManager* s_manager( new PythonManager() );
      return PythonManagerHandle( *s_manager);
   }
}

#endif //Python_Manager_h
