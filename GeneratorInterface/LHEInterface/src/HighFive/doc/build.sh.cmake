#!/bin/bash

# This script generates HTML documentation of the project:
# - The top-level index.html is built from README.md
# - src/examples/index.html is a listing of C++ code snippets
# - HTML snippets are created for every src/examples/*.cpp file

PANDOC=@PANDOC_EXECUTABLE@
if [ x$PANDOC = @PANDOC_EXECUTABLE@ ] ; then
	pandoc = pandoc
fi
examples_dir=@CMAKE_CURRENT_BINARY_DIR@/src/examples

rm -f @CMAKE_CURRENT_BINARY_DIR@/examples.text
rm -rf @CMAKE_CURRENT_BINARY_DIR@/src

cat <<EOF > examples.text
# HighFive C++ Examples
EOF

mkdir -p @CMAKE_CURRENT_BINARY_DIR@/src/examples
for example in @CMAKE_CURRENT_SOURCE_DIR@/../src/examples/*.cpp ; do
	# Generate Markdown snippet from C++ snippet
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {.cpp .numberLines}" > $examples_dir/${example##*/}.text
	cat $example >> $examples_dir/${example##*/}.text
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ " >> $examples_dir/${example##*/}.text
	# Build HTML from Markdown
	$PANDOC \
		$examples_dir/${example##*/}.text \
		-s --highlight-style pygments \
	    -o $examples_dir/${example##*/}.html
	# Get rid of the Markdown
    rm $examples_dir/${example##*/}.text

   	# Extract snippet title: the comments before the `main` function
	comment=`sed -n '/int main/q;p' $example | tac | sed -n '/^\//p; /^[^/]/{q}' | tac | sed '/^\/\/$/d ; s@^// @@' | tr '\n' ' '`
	if [ "x$comment" = x ] ;then
		comment='description is missing'
	fi
	# Reference the snippet
	echo "- [$comment](${example##*/}.html)" >> @CMAKE_CURRENT_BINARY_DIR@/examples.text
done

# src/examples/index.html listing C++ snippets
$PANDOC \
    -c @CMAKE_CURRENT_SOURCE_DIR@/github-pandoc.css examples.text \
    -o $examples_dir/index.html

# Top level index.html
$PANDOC -s -S \
    -c @CMAKE_CURRENT_SOURCE_DIR@/github-pandoc.css \
    @PROJECT_SOURCE_DIR@/README.md \
    -o @CMAKE_CURRENT_BINARY_DIR@/index.html

# Fix links in top-level index.html
# - Insert ".html" to every links to CPP inside <body>...</body>
# - Insert "index.html" to the src/examples/ link inside <body>...</body>
sed '/^<body>/,/<\/body>/ { s/href="\([^"]*cpp\)"/href="\1.html"/ ; s/href="\([^"]*\/\)"/href="\1index.html"/ }' \
    -i @CMAKE_CURRENT_BINARY_DIR@/index.html
