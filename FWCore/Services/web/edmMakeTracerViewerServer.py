import os
import shutil
import argparse

def copy_directory_content(src, dest):
    """
    Copies all content from the source directory to the destination directory.
    """
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dest_path = os.path.join(dest, item)
        
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

def main():
    parser = argparse.ArgumentParser(description='Copy contents of a directory to a specified target directory.')
    parser.add_argument('target_directory', type=str, help='The target directory where content will be copied.')
    
    args = parser.parse_args()
    
    current_directory = os.path.dirname(os.path.abspath(__file__))  # Get the directory where the script is located
    target_directory = args.target_directory
    
    copy_directory_content(current_directory, target_directory)
    print(f"All content from {current_directory} has been copied to {target_directory}")

if __name__ == "__main__":
    main()
