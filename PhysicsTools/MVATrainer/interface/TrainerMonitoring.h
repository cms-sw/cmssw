#ifndef PhysicsTools_MVATrainer_TrainerMonitoring_h
#define PhysicsTools_MVATrainer_TrainerMonitoring_h

#include <string>
#include <vector>
#include <memory>
#include <map>

#include <boost/shared_ptr.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>

#include <TFile.h>
#include <TTree.h>
#include <TClass.h>
#include <TDirectory.h>
#include <TObject.h>

namespace PhysicsTools {

class TrainerMonitoring {
    public:
	TrainerMonitoring(const std::string &fileName);
	~TrainerMonitoring();

	void write();
	void close();

    public:
	class Module;

    protected:
	friend class Module;

	class Object;

	template<typename T>
	class RootObject;

    public:
	class Module {
	    public:
		~Module();

		template<typename T>
		void book(const std::string &name, T *object)
		{ add(new RootObject<T>(name, object)); }

		template<typename T>
		T *book(const std::string &name) {
			T *obj = new T();
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1>
		T *book(const std::string &name, T1 a1) {
			T *obj = new T(a1);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2>
		T *book(const std::string &name, T1 a1, T2 a2) {
			T *obj = new T(a1, a2);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3) {
			T *obj = new T(a1, a2, a3);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4) {
			T *obj = new T(a1, a2, a3, a4);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4, typename T5>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5) {
			T *obj = new T(a1, a2, a3, a4, a5);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4, typename T5, typename T6>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6) {
			T *obj = new T(a1, a2, a3, a4, a5, a6);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4, typename T5, typename T6, typename T7>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7) {
			T *obj = new T(a1, a2, a3, a4, a5, a6, a7);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4, typename T5, typename T6, typename T7,
		         typename T8>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8) {
			T *obj = new T(a1, a2, a3, a4, a5, a6, a7, a8);
			this->reg(name, obj);
			return obj;
		}

		template<typename T, typename T1, typename T2, typename T3,
		         typename T4, typename T5, typename T6, typename T7,
		         typename T8, typename T9>
		T *book(const std::string &name, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, T6 a6, T7 a7, T8 a8, T9 a9) {
			T *obj = new T(a1, a2, a3, a4, a5, a6, a7, a8, a9);
			this->reg(name, obj);
			return obj;
		}

	    protected:
		friend class TrainerMonitoring;

		Module();

		void write(TDirectory *dir);

	    private:
		void add(Object *object);

		template<typename T>
		inline void reg(const std::string &name, T *object);

		TDirectory						*dir;
		std::map<std::string, boost::shared_ptr<Object> >	data;
	};

	Module *book(const std::string &name);

    protected:
	class Object {
	    public:
		Object(const std::string &name) : name(name) {}
		virtual ~Object() {}

		const std::string &getName() const { return name; }

		virtual void write(TDirectory *dir) = 0;

	    private:
		std::string	name;
	};

	template<typename T>
	class RootObject : public Object {
	    public:
		RootObject(const std::string &name, T *object) :
			Object(name), object(object) {}
		virtual ~RootObject() {}

		virtual void write(TDirectory *dir)
		{
			dir->WriteObjectAny(object.get(),
			                    TClass::GetClass(typeid(T)),
			                    getName().c_str());
		}

	    private:
		std::auto_ptr<T>	object;
	};

    private:
	std::auto_ptr<TFile>					rootFile;
	std::map<std::string, boost::shared_ptr<Module> >	modules;
};

namespace helper {
	template<typename T, bool B>
	void trainerMonitoringRootClear(T *object, const boost::mpl::bool_<B>&)
	{}

	template<typename T>
	void trainerMonitoringRootClear(T *object, const boost::mpl::true_&)
	{ object->SetDirectory(0); }
}

template<typename T>
inline void TrainerMonitoring::Module::reg(const std::string &name, T *object)
{
	helper::trainerMonitoringRootClear(object,
	                boost::mpl::or_<
	                	boost::is_base_of<TH1, T>,
	                	boost::is_base_of<TTree, T> >());
	this->book(name, object);
}

} // namespace PhysicsTools

#endif // PhysicsTools_MVATrainer_TrainerMonitoring_h
