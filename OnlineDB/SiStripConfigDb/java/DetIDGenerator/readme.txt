Program used to extract the position of the detectors (det_id) from the construction database ans export the data to the configuration database.

You will need:
	-Java JDK 1.5.*
	-Ant

To compile :
In the DetIDGenerator directory type : ant
That should build the ./bin/detidgenerator.jar file

To run :
You can use the ./generate script in the DetIdGenerator directory
Options :
        -h : This text.
        -mtcc : Informations for the magnet test.
        -export : Export informations to the database defined in $CONFDB.
        -verbose : Display additional informations during the process.

Documentation :
You can type 'ant doc' to generate the code documentation in the ./docs directory

Comments:
For comments, questions, bugs, ... please send a mail to Guillaume Baulieu (g.baulieu@ipnl.in2p3.fr)
