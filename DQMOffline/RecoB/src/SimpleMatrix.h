#ifndef RecoBTag_Analysis_SimpleMatrix_h
#define RecoBTag_Analysis_SimpleMatrix_h

#include <memory>

namespace btag {

template<typename T>
class SimpleMatrix {
    public:
	typedef T					value_type;
	typedef typename std::vector<T>::size_type	size_type;

	SimpleMatrix(size_type rows, size_type cols) :
		width(cols), height(rows), container(rows * cols) {}

	~SimpleMatrix() {}

	inline size_type rows() const { return height; }
	inline size_type cols() const { return width; }
	inline size_type size() const { return container.size(); }

	inline double &operator () (size_type row, size_type col)
	{ return container[index(row, col)]; }
	inline double operator () (size_type row, size_type col) const
	{ return container[index(row, col)]; }

	inline double &operator [] (size_type index)
	{ return container[index]; }
	inline double operator [] (size_type index) const
	{ return container[index]; }

	inline size_type row(size_type index) const { return index / width; }
	inline size_type col(size_type index) const { return index % width; }

    protected:
	size_type index(size_type row, size_type col) const
	{ return row * width + col; }

    private:
	size_type	width, height;
	std::vector<T>	container;
};

} // namespace btag

#endif // GeneratorEvent_Analysis_SimpleMatrix_h
