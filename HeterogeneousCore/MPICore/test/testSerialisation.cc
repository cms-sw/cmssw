// C++ headers
#include <iomanip>
#include <ios>
#include <iostream>
#include <vector>

// ROOT headers
#include <TBufferFile.h>
#include <TClass.h>

template <typename T>
void print_vector(std::vector<T> const& v) {
  if (v.empty()) {
    std::cout << "{}";
    return;
  }

  std::cout << "{ " << v[0];
  for (unsigned int i = 1; i < v.size(); ++i) {
    std::cout << ", " << v[i];
  }
  std::cout << " }";
}

void print_buffer(const char* buffer, int size) {
  auto flags = std::cout.flags();
  for (int i = 0; i < size; ++i) {
    if (i % 16 == 0)
      std::cout << '\t';
    unsigned char value = buffer[i];
    std::cout << "0x" << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)value;
    std::cout << ((i % 16 == 15 or i == size - 1) ? '\n' : ' ');
  }
  std::cout.flags(flags);
}

int main() {
  // Type of the object to serialise and deserialise
  using Type = std::vector<float>;

  // Original vector to serialize
  Type send_object = {1.1, 2.2, 3.3, 4.4, 5.5};

  // Display the contents of the original vector
  std::cout << "Original object:     ";
  print_vector(send_object);
  std::cout << "\n";

  // Create a buffer for serialization
  TBufferFile send_buffer(TBuffer::kWrite);

  // Get the TClass for the type to serialise
  //TClass* type = TClass::GetClass<Type>();
  TClass* type = TClass::GetClass(typeid(Type));

  // Serialize the vector into the buffer
  send_buffer.WriteObjectAny((void*)&send_object, type, false);
  int size = send_buffer.Length();

  // Display the contents of the buffer
  std::cout << "Serialised object is " << size << " bytes long:\n";
  print_buffer(send_buffer.Buffer(), size);

  // Create a new buffer for deserialization
  TBufferFile recv_buffer(TBuffer::kRead, size);

  // Copy the buffer
  std::memcpy(recv_buffer.Buffer(), send_buffer.Buffer(), size);

  // Deserialize into a new vector
  std::unique_ptr<Type> recv_object{reinterpret_cast<Type*>(recv_buffer.ReadObjectAny(type))};

  // Display the contents of the new vector
  std::cout << "Deserialized object: ";
  print_vector(*recv_object);
  std::cout << "\n";

  return 0;
}
