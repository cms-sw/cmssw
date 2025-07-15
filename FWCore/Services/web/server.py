import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
import mimetypes
import argparse
import json

class Serv(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Default route to serve the index.html file
        if self.path == '/':
            self.path = '/index.html'
        elif self.path == '/list-json':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            json_files = sorted([f for f in os.listdir('.') if f.endswith('.json')])
            self.wfile.write(json.dumps(json_files).encode())
            return

        # Serve the requested file (JSON or other static files)
        file_path = self.path[1:]  # Remove leading '/' to get the file path

        try:
            # Read the requested file
            with open(file_path, 'rb') as file:
                file_to_open = file.read()

            # MIME type of the file
            mime_type, _ = mimetypes.guess_type(file_path)

            # Send the HTTP response
            self.send_response(200)
            if mime_type:
                self.send_header("Content-type", mime_type)
            self.end_headers()

            # Write the file content to the response
            self.wfile.write(file_to_open)

        except FileNotFoundError:
            # Handle file not found error
            self.send_response(404)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes("File not found", 'utf-8'))

        except Exception as e:
            # Handle any other internal server errors
            self.send_response(500)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(f"Internal server error: {str(e)}", 'utf-8'))

def run(server_class=HTTPServer, handler_class=Serv, port=65432):
    # Configure and start the HTTP server
    server_address = ('localhost', port)
    httpd = server_class(server_address, handler_class)
    print(f"Server started at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    # Parse command-line arguments to set the server port
    parser = argparse.ArgumentParser(description='Start a simple HTTP server.')
    parser.add_argument('--port', type=int, default=65432, help='Port to serve on (default: 65432)')
    args = parser.parse_args()

    # Start the server with the specified port
    run(port=args.port)



