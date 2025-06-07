#!/bin/bash
# Script to rename AOT products from temporary names

# Check for directory argument
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

TARGET_DIR="$1"

# Make sure the directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

# Go through files in the directory
for file in "$TARGET_DIR"/*; do
  filename=$(basename "$file")

  case "$filename" in
    *_metadata.json)
      mv "$file" "$TARGET_DIR/model_metadata.json"
      ;;
    *_compile_flags.json)
      mv "$file" "$TARGET_DIR/model_compile_flags.json"
      ;;
    *_linker_flags.json)
      mv "$file" "$TARGET_DIR/model_linker_flags.json"
      ;;  
    *.cpp)
      mv "$file" "$TARGET_DIR/model.cpp"
      ;;
    *.o)
      mv "$file" "$TARGET_DIR/external.o"
      ;;
  esac
done
