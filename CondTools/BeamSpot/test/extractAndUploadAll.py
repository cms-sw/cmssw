import sqlite3
import json
import subprocess
import shutil

# Open the SQLite database file
conn = sqlite3.connect('test_myTagName.db')
cursor = conn.cursor()

# Execute the SELECT query
cursor.execute('SELECT * FROM IOV')

# Fetch all the results from the query
results = cursor.fetchall()

# Create an empty list to store the extracted values
extracted_values = []

# Extract the desired part of the string and add it to the list
for row in results:
    value = row[0].split('|')[0].strip()
    extracted_values.append(value)

# Close the database connection
conn.close()

# Generate the JSON file and execute the command for each extracted value
for value in extracted_values:
    print("uploading",value)
    # Create the dictionary for the JSON structure
    data = {
        "destinationDatabase": "oracle://cms_orcoff_prep/CMS_CONDITIONS",
        "destinationTags": {
            "SimBeamSpot_" + value + "_v0_mc": {}
        },
        "inputTag": value,
        "since": None,
        "userText": value
    }
    
    # Generate the JSON file
    filename = "test_myTagName.txt"
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    # Execute the command to upload conditions
    subprocess.call(["uploadConditions.py", "test_myTagName.db"])

    # Generate the new filename
    new_filename = "test_" + value + ".txt"
    
    # Move the file to the new name
    shutil.move(filename, new_filename)

    # Print a success message
    print("Uploaded conditions for value:", value)
