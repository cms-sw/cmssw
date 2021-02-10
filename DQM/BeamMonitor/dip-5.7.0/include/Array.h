#ifndef ARRAY_H
#define ARRAY_H

#include "platformDependantOptions.h"
/**
* I've been having real problems exporting STL containers from shared libraries (particularly in Windows). Hence I'm having ago at writing 
* my own containers
*/

template <class T> class SArray;

template <class T> class SList{
private:
	friend class SArray<T>;
	struct ListElement{
		ListElement *prev;
		ListElement *next;
		T data;

		ListElement(T d):prev(NULL),next(NULL),data(d){};
	};

	unsigned noElements;

	ListElement * start;

	ListElement * end;

public:
	class Iterator{
	private:
		friend class SList<T>;
		friend class SArray<T>;
		ListElement * current;
	public:
		Iterator(ListElement *element):current(element){};

		Iterator():current(NULL){};

		bool isValid(){return (current != NULL);}

		Iterator & operator++(){
			current = current->next;
			return *this;
		}


		Iterator & operator--(){
			current = current->prev;
			return *this;
		}


		Iterator & operator=(Iterator &i){
			current = i.current;
			return *this;
		}


		T operator->(){
			return current->data;			
		}

		T operator*(){
			return current->data;			
		}
	};


	SList():noElements(0), start(NULL), end(NULL){};

	~SList(){
		while (start){
			ListElement * tmp = start;
			start = start->next;
			delete tmp;
		}
	}

	Iterator addToBack(T data){
		ListElement * element = new ListElement(data);
		if (!end){
			start = end = element;
		} else {
			element->prev = end;
			end->next = element;
			end = element;
		}
		noElements++;
		return Iterator(element);
	}


	Iterator addBefore(Iterator &i, T data){
		ListElement * element = new ListElement(data);
		element->next = i.current;
		element->prev = i.current->prev;
		i.current->prev = element;
		if (element->prev){
			element->prev->next = element;
		} else {
			start = element;
		}
		noElements++;
		return Iterator(element);
	}


	void remove(Iterator &i){
		if (i.current->next){
			i.current->next->prev = i.current->prev;
		} else {
			end = i.current->prev;
		}

		if (i.current->prev){
			i.current->prev->next = i.current->next;
		} else {
			start = i.current->next;
		}

		delete i.current;
		noElements--;
	}


	const unsigned size() const{return noElements;}


	Iterator begin(){
		return Iterator(start);
	}
};





template <class T> class SArray
{
private:
	DataBlock dataStore;

	typedef SList<T> List;
	List list;

public:
	SArray(unsigned reserve = 10):dataStore(reserve*sizeof(typename List::Iterator)){};

	void add(T /*&*/element){
		typename List::Iterator i = list.addToBack(element);
		dataStore.write(&i, sizeof(typename List::Iterator), (list.size()-1)*sizeof(typename List::Iterator));
	}


	T & operator[](unsigned index){
		typename List::Iterator i;
		i  = *((typename List::Iterator *) dataStore.read(sizeof(i), index*sizeof(i)));
		return i.current->data;
	}

	const unsigned size() const{
		return list.size();
	}
};

#endif
