template<typename T, unsigned int S>
class fixedArray {
    public:
        fixedArray() {}

        operator T*() { return content; }
        operator const T*() const { return content; }
	const T& operator [](const unsigned int i) { return content[i]; }
    private:
        T content[S];
};
