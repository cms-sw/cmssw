#!/bin/tcsh
cat HELP2.html.beg > HELP2.html


set YEARTAG=2012
echo "<h2> =============  2012 year</h2> \n"  >> HELP2.html
echo "<h2> +++++++++ Runs with more then 10000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

echo "<h2> +++++++++ Runs with more then 5000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

echo "<h2> +++++++++ Runs with more then 2000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

set YEARTAG=2011
echo "<h2> =============  2011 year \n"  >> HELP2.html
echo "<h2> +++++++++ Runs with more then 10000 events </h2>\n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

echo "<h2> +++++++++ Runs with more then 5000 events </h2>\n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h2> +++++++++ Runs with more then 2000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html 
cat run2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html 
echo "<h3>"  >> HELP2.html
cat days2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h2> +++++++++ Runs with more then 2000 events and less then 10000 events </h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat run2000_10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days2000_10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

set YEARTAG=2010
echo "<h2> =============  2010 year</h2> \n"  >> HELP2.html
echo "<h2> +++++++++ Runs with more then 10000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html 
cat run10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days10000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

echo "<h2> +++++++++ Runs with more then 5000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html 
cat run5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days5000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

echo "<h2> +++++++++ Runs with more then 2000 events</h2> \n"  >> HELP2.html
echo "<h3>"  >> HELP2.html 
cat run2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html
echo "<h3>"  >> HELP2.html
cat days2000_${YEARTAG}.txt1 >> HELP2.html
echo "</h3>"  >> HELP2.html

cat HELP2.html.end >> HELP2.html
